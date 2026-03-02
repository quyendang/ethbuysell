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

# -----------------------------
# Config
# -----------------------------
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
EXCHANGE_NAME = os.getenv("EXCHANGE", "binance").strip()
SYMBOL = os.getenv("SYMBOL", "ETH/USDT").strip()
CHECK_INTERVAL_SEC = int(os.getenv("CHECK_INTERVAL_SEC", "60"))

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in .env")

TF_REGIME = "4h"   # H4
TF_ENTRY = "1h"    # H1

# Anti-miss thresholds (tune-able)
ANTI_MISS_RSI_H4 = float(os.getenv("ANTI_MISS_RSI_H4", "55"))
ANTI_MISS_FLAT_SLOPE_PCT = float(os.getenv("ANTI_MISS_FLAT_SLOPE_PCT", "0.002"))  # 0.2% per bar
ANTI_MISS_DIST_SHRINK_BARS = int(os.getenv("ANTI_MISS_DIST_SHRINK_BARS", "6"))    # last 6 H4 candles

# -----------------------------
# Telegram
# -----------------------------
def tg_send(text: str) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=15) as r:
        r.read()

# -----------------------------
# Indicators
# -----------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def slope_last(series: pd.Series, lookback: int = 12) -> float:
    s = series.dropna().tail(lookback)
    if len(s) < lookback:
        return 0.0
    y = s.values
    x = np.arange(len(y))
    m = np.polyfit(x, y, 1)[0]
    return float(m)

# -----------------------------
# Candlestick patterns
# -----------------------------
def bearish_engulfing(df: pd.DataFrame) -> bool:
    if len(df) < 3:
        return False
    prev = df.iloc[-2]
    last = df.iloc[-1]
    prev_body_low = min(prev["open"], prev["close"])
    prev_body_high = max(prev["open"], prev["close"])
    last_body_low = min(last["open"], last["close"])
    last_body_high = max(last["open"], last["close"])
    return (prev["close"] > prev["open"]) and (last["close"] < last["open"]) and \
           (last_body_low <= prev_body_low) and (last_body_high >= prev_body_high)

def bullish_engulfing(df: pd.DataFrame) -> bool:
    if len(df) < 3:
        return False
    prev = df.iloc[-2]
    last = df.iloc[-1]
    prev_body_low = min(prev["open"], prev["close"])
    prev_body_high = max(prev["open"], prev["close"])
    last_body_low = min(last["open"], last["close"])
    last_body_high = max(last["open"], last["close"])
    return (prev["close"] < prev["open"]) and (last["close"] > last["open"]) and \
           (last_body_low <= prev_body_low) and (last_body_high >= prev_body_high)

def rejection_wick(df: pd.DataFrame, wick_ratio: float = 1.8) -> bool:
    if len(df) < 2:
        return False
    last = df.iloc[-1]
    o, c, h, l = last["open"], last["close"], last["high"], last["low"]
    body = abs(c - o)
    if body == 0:
        body = (h - l) * 0.1
    upper_wick = h - max(o, c)
    return upper_wick >= wick_ratio * body

# -----------------------------
# Exchange data
# -----------------------------
def make_exchange():
    ex_class = getattr(ccxt, EXCHANGE_NAME, None)
    if ex_class is None:
        raise RuntimeError(f"Unsupported exchange: {EXCHANGE_NAME}")
    return ex_class({"enableRateLimit": True})

def fetch_ohlcv(ex, symbol: str, timeframe: str, limit: int = 300) -> pd.DataFrame:
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("ts", inplace=True)
    return df

# -----------------------------
# Strategy logic
# -----------------------------
@dataclass
class RegimeResult:
    ok: bool
    score: int
    details: Dict[str, bool]
    ema200_slope: float

def compute_regime(df_h4: pd.DataFrame, df_btc_h4: pd.DataFrame) -> RegimeResult:
    c = df_h4["close"]
    ema50 = ema(c, 50)
    ema200 = ema(c, 200)
    ema200_sl = slope_last(ema200, lookback=12)

    cond1 = bool(ema50.iloc[-1] < ema200.iloc[-1])
    cond2 = bool(ema200_sl < 0)
    cond3 = bool(c.iloc[-1] < ema200.iloc[-1])

    btc_c = df_btc_h4["close"]
    btc_ema50 = ema(btc_c, 50)
    btc_ema200 = ema(btc_c, 200)
    cond4 = bool(btc_ema50.iloc[-1] < btc_ema200.iloc[-1])

    details = {
        "EMA50_lt_EMA200": cond1,
        "EMA200_slope_down": cond2,
        "Price_lt_EMA200": cond3,
        "BTC_confirm": cond4,
    }
    score = sum(details.values())
    ok = score >= 4
    return RegimeResult(ok=ok, score=score, details=details, ema200_slope=ema200_sl)

@dataclass
class AntiMissResult:
    stop_sell_mode: bool
    reason: str

def compute_anti_miss(df_h4: pd.DataFrame) -> AntiMissResult:
    c = df_h4["close"]
    ema50 = ema(c, 50)
    ema200 = ema(c, 200)
    rsi_h4 = rsi(c, 14)

    close_now = float(c.iloc[-1])
    ema200_now = float(ema200.iloc[-1])

    # (A) Strong reversal: close above EMA200
    if close_now > ema200_now:
        return AntiMissResult(
            stop_sell_mode=True,
            reason=f"Close(H4) {close_now:.2f} > EMA200(H4) {ema200_now:.2f} (reversal risk)"
        )

    # (B) Weakening downtrend
    ema200_sl = slope_last(ema200, lookback=12)
    ema200_slope_pct = abs(ema200_sl) / close_now if close_now > 0 else 0.0
    flat = ema200_slope_pct <= (ANTI_MISS_FLAT_SLOPE_PCT / 100.0)

    # distance shrinking: (EMA200-EMA50) decreasing over last N bars
    dist = (ema200 - ema50).dropna().tail(ANTI_MISS_DIST_SHRINK_BARS + 1)
    shrink = False
    if len(dist) >= ANTI_MISS_DIST_SHRINK_BARS + 1:
        shrink = float(dist.iloc[-1]) < float(dist.iloc[0])

    rsi_ok = float(rsi_h4.iloc[-1]) > ANTI_MISS_RSI_H4

    if flat and shrink and rsi_ok:
        return AntiMissResult(
            stop_sell_mode=True,
            reason=(
                f"Downtrend weakening: EMA200 slope ~flat ({ema200_slope_pct*100:.4f}%/bar), "
                f"dist(EMA200-EMA50) shrinking, RSI(H4)={float(rsi_h4.iloc[-1]):.1f} > {ANTI_MISS_RSI_H4}"
            )
        )

    return AntiMissResult(stop_sell_mode=False, reason="")

def recent_swing_low(df: pd.DataFrame, lookback: int = 72) -> Tuple[float, pd.Timestamp]:
    sub = df.tail(lookback)
    idx = sub["low"].idxmin()
    return float(sub.loc[idx, "low"]), idx

def volume_is_fading(df: pd.DataFrame, bars: int = 12) -> bool:
    v = df["volume"].tail(bars).values
    if len(v) < bars:
        return False
    x = np.arange(len(v))
    m = np.polyfit(x, v, 1)[0]
    return m < 0

def fib_zone_from_swing(df: pd.DataFrame, lookback: int = 96) -> Tuple[float, float]:
    sub = df.tail(lookback)
    swing_low = sub["low"].min()
    swing_high = sub["high"].max()
    if swing_high <= swing_low:
        return (math.nan, math.nan)
    move = swing_high - swing_low
    z618 = swing_high - 0.618 * move
    z50 = swing_high - 0.5 * move
    lo = min(z50, z618)
    hi = max(z50, z618)
    return float(lo), float(hi)

@dataclass
class Signal:
    type: str  # "SELL" | "BUY" | "NONE"
    reason: str
    price: float

def generate_signals(df_h4: pd.DataFrame, df_h1: pd.DataFrame, regime: RegimeResult, anti: AntiMissResult) -> Signal:
    c1 = df_h1["close"]
    ema50_h1 = ema(c1, 50)
    ema50_h4 = float(ema(df_h4["close"], 50).iloc[-1])
    rsi_h1 = rsi(c1, 14)
    atr_h1 = atr(df_h1, 14)

    price = float(c1.iloc[-1])
    rsi_now = float(rsi_h1.iloc[-1])
    atr_now = float(atr_h1.iloc[-1]) if not np.isnan(atr_h1.iloc[-1]) else 0.0

    # ---- SELL (only if regime ok AND NOT stop-sell)
    if regime.ok and (not anti.stop_sell_mode):
        low_val, _ = recent_swing_low(df_h1, lookback=72)
        retrace_pct = (price - low_val) / low_val * 100 if low_val > 0 else 0

        near_ema50_h1 = abs(price - float(ema50_h1.iloc[-1])) <= (0.35 * atr_now) if atr_now > 0 else False
        near_ema50_h4 = abs(price - float(ema50_h4)) <= (0.6 * atr_now) if atr_now > 0 else False
        fib_lo, fib_hi = fib_zone_from_swing(df_h1, lookback=96)
        in_fib_zone = (not np.isnan(fib_lo)) and (fib_lo <= price <= fib_hi)

        setup_ok = (4.0 <= retrace_pct <= 12.0) and (rsi_now > 62.0) and \
                   (near_ema50_h1 or near_ema50_h4 or in_fib_zone) and volume_is_fading(df_h1, bars=12)

        trigger_ok = bearish_engulfing(df_h1) or rejection_wick(df_h1, wick_ratio=1.8)

        if setup_ok and trigger_ok:
            reason = (
                f"RegimeScore={regime.score}/4, Retrace={retrace_pct:.1f}%, RSI(H1)={rsi_now:.1f}, "
                f"Zone={'EMA50H1' if near_ema50_h1 else ('EMA50H4' if near_ema50_h4 else 'FIB')}, "
                f"Trigger={'BearEngulf' if bearish_engulfing(df_h1) else 'RejectionWick'}"
            )
            return Signal(type="SELL", reason=reason, price=price)

    # ---- BUY (works anytime; best when you've sold before)
    rsi_buy = rsi_now < 38.0
    last_high = float(df_h1.tail(24)["high"].max())
    drop_pct = (last_high - price) / last_high * 100 if last_high > 0 else 0
    drop_buy = drop_pct >= 6.0

    trigger_buy = bullish_engulfing(df_h1) or (rsi_buy and (not bearish_engulfing(df_h1)))
    if (rsi_buy or drop_buy) and trigger_buy:
        reason = f"RSI(H1)={rsi_now:.1f}, DropFrom24hHigh={drop_pct:.1f}%, Trigger={'BullEngulf' if bullish_engulfing(df_h1) else 'Confirm'}"
        return Signal(type="BUY", reason=reason, price=price)

    return Signal(type="NONE", reason="", price=price)

# -----------------------------
# State
# -----------------------------
STATE_FILE = "bot_state.json"

def load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {"last_sell_ts": None, "last_buy_ts": None, "stop_sell_mode": False, "last_mode_ts": None}
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_state(state: dict) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def can_notify_once_per_h1(state: dict, sig_type: str, bar_ts: pd.Timestamp) -> bool:
    key = "last_sell_ts" if sig_type == "SELL" else "last_buy_ts"
    last = state.get(key)
    return last != bar_ts.isoformat()

def mark_notified(state: dict, sig_type: str, bar_ts: pd.Timestamp) -> None:
    key = "last_sell_ts" if sig_type == "SELL" else "last_buy_ts"
    state[key] = bar_ts.isoformat()

def mode_changed(state: dict, new_mode: bool, ts: pd.Timestamp) -> bool:
    if bool(state.get("stop_sell_mode", False)) != bool(new_mode):
        # ensure we don't spam on same H4 bar
        last_mode_ts = state.get("last_mode_ts")
        return last_mode_ts != ts.isoformat()
    return False

# -----------------------------
# Main
# -----------------------------
def main():
    ex = make_exchange()
    state = load_state()

    tg_send(
        f"✅ Spot Alert Bot started\n"
        f"Symbol: {SYMBOL}\nExchange: {EXCHANGE_NAME}\nRegime: {TF_REGIME}, Entry: {TF_ENTRY}\n"
        f"Anti-miss: RSI_H4>{ANTI_MISS_RSI_H4}, flatSlope<={ANTI_MISS_FLAT_SLOPE_PCT}%/bar"
    )

    while True:
        try:
            df_h4 = fetch_ohlcv(ex, SYMBOL, TF_REGIME, limit=300)
            df_h1 = fetch_ohlcv(ex, SYMBOL, TF_ENTRY, limit=400)

            btc_symbol = "BTC/USDT"
            df_btc_h4 = fetch_ohlcv(ex, btc_symbol, TF_REGIME, limit=300)

            regime = compute_regime(df_h4, df_btc_h4)
            anti = compute_anti_miss(df_h4)

            h4_ts = df_h4.index[-1]
            if mode_changed(state, anti.stop_sell_mode, h4_ts):
                if anti.stop_sell_mode:
                    tg_send(f"🟠 <b>STOP SELL MODE</b>\nTime(UTC): {h4_ts.strftime('%Y-%m-%d %H:%M')}\nReason: {anti.reason}\n"
                            f"Note: Bot sẽ tạm ngừng gửi SELL signal để tránh bán rồi bị chạy mất trend.")
                else:
                    tg_send(f"🟢 <b>SELL MODE RESUMED</b>\nTime(UTC): {h4_ts.strftime('%Y-%m-%d %H:%M')}\n"
                            f"Note: Điều kiện anti-miss đã hết, bot tiếp tục lọc SELL theo downtrend.")

                state["stop_sell_mode"] = anti.stop_sell_mode
                state["last_mode_ts"] = h4_ts.isoformat()
                save_state(state)

            sig = generate_signals(df_h4, df_h1, regime, anti)
            h1_ts = df_h1.index[-1]

            if sig.type in ("SELL", "BUY") and can_notify_once_per_h1(state, sig.type, h1_ts):
                d = regime.details
                regime_txt = (
                    f"Regime(H4) Score: {regime.score}/4\n"
                    f"- EMA50<EMA200: {d['EMA50_lt_EMA200']}\n"
                    f"- EMA200 slope down: {d['EMA200_slope_down']} (slope={regime.ema200_slope:.6f})\n"
                    f"- Price<EMA200: {d['Price_lt_EMA200']}\n"
                    f"- BTC confirm: {d['BTC_confirm']}\n"
                    f"- STOP_SELL_MODE: {anti.stop_sell_mode}\n"
                )

                tg_send(
                    f"🚨 <b>{sig.type} SIGNAL</b> ({SYMBOL})\n"
                    f"Time (UTC): {h1_ts.strftime('%Y-%m-%d %H:%M')}\n"
                    f"Price: <b>{sig.price:.2f}</b>\n\n"
                    f"{regime_txt}\n"
                    f"Reason: {sig.reason}\n\n"
                    f"Note: Bot chỉ cảnh báo spot (không đặt lệnh)."
                )

                mark_notified(state, sig.type, h1_ts)
                save_state(state)

            time.sleep(CHECK_INTERVAL_SEC)

        except Exception as e:
            try:
                tg_send(f"⚠️ Bot error: {type(e).__name__}: {e}")
            except Exception:
                pass
            time.sleep(max(10, CHECK_INTERVAL_SEC))

if __name__ == "__main__":
    main()