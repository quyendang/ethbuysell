import os
import time
import json
import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import ccxt
from dotenv import load_dotenv
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# ==============================
# CONFIG
# ==============================
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
EXCHANGE_NAME = os.getenv("EXCHANGE", "binance").strip()
SYMBOL = os.getenv("SYMBOL", "ETH/USDT").strip()
CHECK_INTERVAL_SEC = int(os.getenv("CHECK_INTERVAL_SEC", "60"))

TF_REGIME = "4h"
TF_ENTRY = "1h"

ANTI_MISS_RSI_H4 = 55
ANTI_MISS_DIST_BARS = 6

# ==============================
# TELEGRAM SAFE SEND
# ==============================
def tg_send(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    text = (text or "").strip()
    if len(text) > 3900:
        text = text[:3900] + "\n...(truncated)"

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True,
    }

    try:
        req = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=15) as r:
            r.read()
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        print(f"[TELEGRAM ERROR] HTTP {e.code}: {body}")
    except URLError as e:
        print(f"[TELEGRAM ERROR] URLError: {e}")
    except Exception as e:
        print(f"[TELEGRAM ERROR] {e}")

# ==============================
# INDICATORS
# ==============================
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def slope(series, lookback=12):
    s = series.dropna().tail(lookback)
    if len(s) < lookback:
        return 0
    y = s.values
    x = np.arange(len(y))
    return np.polyfit(x, y, 1)[0]

# ==============================
# EXCHANGE
# ==============================
def make_exchange():
    return getattr(ccxt, EXCHANGE_NAME)({"enableRateLimit": True})

def fetch_df(ex, symbol, tf, limit=300):
    data = ex.fetch_ohlcv(symbol, tf, limit=limit)
    df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("ts", inplace=True)
    return df

# ==============================
# REGIME + ANTI MISS
# ==============================
@dataclass
class Regime:
    ok: bool
    score: int
    ema200_slope: float

def compute_regime(df_h4, df_btc):
    c = df_h4["close"]
    ema50 = ema(c, 50)
    ema200 = ema(c, 200)

    btc_ema50 = ema(df_btc["close"], 50)
    btc_ema200 = ema(df_btc["close"], 200)

    cond1 = ema50.iloc[-1] < ema200.iloc[-1]
    cond2 = slope(ema200) < 0
    cond3 = c.iloc[-1] < ema200.iloc[-1]
    cond4 = btc_ema50.iloc[-1] < btc_ema200.iloc[-1]

    score = sum([cond1, cond2, cond3, cond4])
    return Regime(ok=score >= 4, score=score, ema200_slope=slope(ema200))

def anti_miss(df_h4):
    c = df_h4["close"]
    ema50 = ema(c, 50)
    ema200 = ema(c, 200)
    rsi_h4 = rsi(c)

    close_now = c.iloc[-1]
    ema200_now = ema200.iloc[-1]

    if close_now > ema200_now:
        return True, "Close vượt EMA200"

    dist = (ema200 - ema50).tail(ANTI_MISS_DIST_BARS+1)
    shrinking = dist.iloc[-1] < dist.iloc[0]

    if shrinking and rsi_h4.iloc[-1] > ANTI_MISS_RSI_H4:
        return True, "Downtrend yếu dần + RSI H4 tăng"

    return False, ""

# ==============================
# SIGNAL LOGIC
# ==============================
def check_sell(df_h4, df_h1, regime, stop_mode):
    if not regime.ok or stop_mode:
        return False, ""

    c = df_h1["close"]
    ema50_h1 = ema(c, 50)
    rsi_h1 = rsi(c)

    price = c.iloc[-1]
    retrace = (price - df_h1["low"].tail(72).min()) / df_h1["low"].tail(72).min() * 100

    if retrace > 4 and rsi_h1.iloc[-1] > 62 and price >= ema50_h1.iloc[-1]:
        return True, f"Retrace {retrace:.1f}% + RSI {rsi_h1.iloc[-1]:.1f}"

    return False, ""

def check_buy(df_h1):
    c = df_h1["close"]
    rsi_h1 = rsi(c)
    price = c.iloc[-1]
    drop = (df_h1["high"].tail(24).max() - price) / df_h1["high"].tail(24).max() * 100

    if rsi_h1.iloc[-1] < 38 or drop >= 6:
        return True, f"RSI {rsi_h1.iloc[-1]:.1f} + Drop {drop:.1f}%"

    return False, ""

# ==============================
# MAIN LOOP
# ==============================
def main():
    ex = make_exchange()

    tg_send(f"Bot started for {SYMBOL}")

    last_signal_time = None
    stop_mode_prev = False

    while True:
        try:
            print("Bot alive:", pd.Timestamp.now('UTC'))
            df_h4 = fetch_df(ex, SYMBOL, TF_REGIME)
            df_h1 = fetch_df(ex, SYMBOL, TF_ENTRY)
            df_btc = fetch_df(ex, "BTC/USDT", TF_REGIME)

            regime = compute_regime(df_h4, df_btc)
            stop_mode, reason = anti_miss(df_h4)

            if stop_mode != stop_mode_prev:
                if stop_mode:
                    tg_send(f"STOP SELL MODE: {reason}")
                else:
                    tg_send("SELL MODE RESUMED")
                stop_mode_prev = stop_mode

            sell, sell_reason = check_sell(df_h4, df_h1, regime, stop_mode)
            buy, buy_reason = check_buy(df_h1)

            now_bar = df_h1.index[-1]

            if sell and last_signal_time != now_bar:
                tg_send(f"SELL SIGNAL {SYMBOL}\nPrice: {df_h1['close'].iloc[-1]:.2f}\n{sell_reason}")
                last_signal_time = now_bar

            elif buy and last_signal_time != now_bar:
                tg_send(f"BUY SIGNAL {SYMBOL}\nPrice: {df_h1['close'].iloc[-1]:.2f}\n{buy_reason}")
                last_signal_time = now_bar

            time.sleep(CHECK_INTERVAL_SEC)

        except Exception as e:
            print(f"[ERROR] {e}")
            time.sleep(15)

if __name__ == "__main__":
    main()