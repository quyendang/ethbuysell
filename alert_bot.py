import os
import time
import json
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Any

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
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()  # for push alerts (optional for /check replies)
EXCHANGE_NAME = os.getenv("EXCHANGE", "binance").strip()
SYMBOL = os.getenv("SYMBOL", "ETH/USDT").strip()
CHECK_INTERVAL_SEC = int(os.getenv("CHECK_INTERVAL_SEC", "60"))

TF_REGIME = "4h"   # H4 regime
TF_ENTRY = "1h"    # H1 timing

# Anti-miss
ANTI_MISS_RSI_H4 = float(os.getenv("ANTI_MISS_RSI_H4", "55"))
ANTI_MISS_DIST_BARS = int(os.getenv("ANTI_MISS_DIST_BARS", "6"))

# Telegram polling
TG_POLL_TIMEOUT_SEC = int(os.getenv("TG_POLL_TIMEOUT_SEC", "0"))  # keep 0 to avoid long blocking in Koyeb loop

STATE_FILE = "bot_state.json"

# ==============================
# TELEGRAM SAFE SEND + REPLY
# ==============================
def tg_api(method: str, payload: dict, timeout: int = 20) -> dict:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as r:
        raw = r.read().decode("utf-8", errors="ignore")
    try:
        return json.loads(raw)
    except Exception:
        return {"ok": False, "raw": raw}

def tg_send(text: str, chat_id: Optional[str] = None) -> None:
    """Send plain text. Never crash the bot if Telegram fails."""
    if not TELEGRAM_BOT_TOKEN:
        print("[TELEGRAM ERROR] Missing TELEGRAM_BOT_TOKEN")
        return

    text = (text or "").strip()
    if len(text) > 3900:
        text = text[:3900] + "\n...(truncated)"

    payload = {
        "chat_id": chat_id or TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True,
    }

    try:
        if not payload["chat_id"]:
            print("[TELEGRAM ERROR] Missing TELEGRAM_CHAT_ID (and no chat_id provided).")
            return
        resp = tg_api("sendMessage", payload, timeout=20)
        if not resp.get("ok", False):
            print(f"[TELEGRAM ERROR] sendMessage failed: {resp}")
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        print(f"[TELEGRAM ERROR] HTTP {e.code}: {body}")
    except URLError as e:
        print(f"[TELEGRAM ERROR] URLError: {e}")
    except Exception as e:
        print(f"[TELEGRAM ERROR] {type(e).__name__}: {e}")

def tg_get_updates(offset: Optional[int]) -> dict:
    """Poll Telegram updates (non-blocking by default)."""
    payload = {"timeout": TG_POLL_TIMEOUT_SEC}
    if offset is not None:
        payload["offset"] = offset
    try:
        return tg_api("getUpdates", payload, timeout=25)
    except Exception as e:
        print(f"[TELEGRAM ERROR] getUpdates: {e}")
        return {"ok": False, "result": []}

# ==============================
# STATE
# ==============================
def load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {
            "last_signal_h1_ts": None,
            "stop_sell_mode": False,
            "last_mode_h4_ts": None,
            "tg_update_offset": None,
        }
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {
            "last_signal_h1_ts": None,
            "stop_sell_mode": False,
            "last_mode_h4_ts": None,
            "tg_update_offset": None,
        }

def save_state(state: dict) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

# ==============================
# INDICATORS
# ==============================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
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

def slope(series: pd.Series, lookback: int = 12) -> float:
    s = series.dropna().tail(lookback)
    if len(s) < lookback:
        return 0.0
    y = s.values
    x = np.arange(len(y))
    return float(np.polyfit(x, y, 1)[0])

# ==============================
# EXCHANGE
# ==============================
def make_exchange():
    ex_cls = getattr(ccxt, EXCHANGE_NAME, None)
    if ex_cls is None:
        raise RuntimeError(f"Unsupported exchange: {EXCHANGE_NAME}")
    return ex_cls({"enableRateLimit": True})

def fetch_df(ex, symbol: str, tf: str, limit: int = 300) -> pd.DataFrame:
    data = ex.fetch_ohlcv(symbol, tf, limit=limit)
    df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "volume"])
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
    details: Dict[str, bool]
    ema200_slope: float

def compute_regime(df_h4: pd.DataFrame, df_btc_h4: pd.DataFrame) -> Regime:
    c = df_h4["close"]
    ema50 = ema(c, 50)
    ema200 = ema(c, 200)

    btc_c = df_btc_h4["close"]
    btc_ema50 = ema(btc_c, 50)
    btc_ema200 = ema(btc_c, 200)

    cond1 = bool(ema50.iloc[-1] < ema200.iloc[-1])
    cond2 = bool(slope(ema200, 12) < 0)
    cond3 = bool(c.iloc[-1] < ema200.iloc[-1])
    cond4 = bool(btc_ema50.iloc[-1] < btc_ema200.iloc[-1])

    details = {
        "EMA50_lt_EMA200": cond1,
        "EMA200_slope_down": cond2,
        "Price_lt_EMA200": cond3,
        "BTC_confirm": cond4
    }
    score = sum(details.values())
    ok = score >= 4
    return Regime(ok=ok, score=score, details=details, ema200_slope=slope(ema200, 12))

def compute_stop_sell_mode(df_h4: pd.DataFrame) -> Tuple[bool, str]:
    c = df_h4["close"]
    ema50 = ema(c, 50)
    ema200 = ema(c, 200)
    rsi_h4 = rsi(c, 14)

    close_now = float(c.iloc[-1])
    ema200_now = float(ema200.iloc[-1])

    # A) strong reversal risk
    if close_now > ema200_now:
        return True, f"Close(H4) {close_now:.2f} > EMA200(H4) {ema200_now:.2f}"

    # B) weakening downtrend (distance shrinking + RSI rising)
    dist = (ema200 - ema50).dropna().tail(ANTI_MISS_DIST_BARS + 1)
    shrinking = False
    if len(dist) >= ANTI_MISS_DIST_BARS + 1:
        shrinking = float(dist.iloc[-1]) < float(dist.iloc[0])

    if shrinking and float(rsi_h4.iloc[-1]) > ANTI_MISS_RSI_H4:
        return True, f"Dist(EMA200-EMA50) shrinking + RSI(H4) {float(rsi_h4.iloc[-1]):.1f} > {ANTI_MISS_RSI_H4}"

    return False, ""

# ==============================
# BUYZONE / SELLZONE (explainable)
# ==============================
def fib_zone_from_swing(df: pd.DataFrame, lookback: int = 96) -> Tuple[float, float]:
    sub = df.tail(lookback)
    swing_low = float(sub["low"].min())
    swing_high = float(sub["high"].max())
    if swing_high <= swing_low:
        return (math.nan, math.nan)
    move = swing_high - swing_low
    z618 = swing_high - 0.618 * move
    z50 = swing_high - 0.5 * move
    lo = min(z50, z618)
    hi = max(z50, z618)
    return float(lo), float(hi)

def compute_zones(df_h4: pd.DataFrame, df_h1: pd.DataFrame) -> Dict[str, Any]:
    """
    Zones based on your conditions:
    - SELL zone: near EMA50(H1) or EMA50(H4) or Fib 0.5-0.618 (H1 swing), when regime down.
    - BUY zone: near recent support (recent swing low area) and RSI low.
    """
    price = float(df_h1["close"].iloc[-1])

    ema50_h1 = float(ema(df_h1["close"], 50).iloc[-1])
    ema200_h1 = float(ema(df_h1["close"], 200).iloc[-1])
    ema50_h4 = float(ema(df_h4["close"], 50).iloc[-1])
    ema200_h4 = float(ema(df_h4["close"], 200).iloc[-1])

    atr_h1 = float(atr(df_h1, 14).iloc[-1]) if not np.isnan(atr(df_h1, 14).iloc[-1]) else 0.0

    fib_lo, fib_hi = fib_zone_from_swing(df_h1, lookback=96)

    # Support proxy: lowest low last 72 H1 bars
    support = float(df_h1["low"].tail(72).min())

    # Suggest a band (zone width uses ATR)
    w_sell = max(atr_h1 * 0.6, price * 0.002)  # ~0.2% min
    w_buy = max(atr_h1 * 0.8, price * 0.002)

    sell_zone_candidates = []
    sell_zone_candidates.append(("EMA50_H1", ema50_h1 - w_sell, ema50_h1 + w_sell))
    sell_zone_candidates.append(("EMA50_H4", ema50_h4 - w_sell, ema50_h4 + w_sell))
    if not np.isnan(fib_lo) and not np.isnan(fib_hi):
        sell_zone_candidates.append(("FIB_0.5_0.618", fib_lo, fib_hi))

    buy_zone = (support - w_buy, support + w_buy)

    return {
        "price": price,
        "ema50_h1": ema50_h1,
        "ema200_h1": ema200_h1,
        "ema50_h4": ema50_h4,
        "ema200_h4": ema200_h4,
        "atr_h1": atr_h1,
        "fib_zone": (fib_lo, fib_hi),
        "support": support,
        "sell_zones": sell_zone_candidates,
        "buy_zone": buy_zone,
    }

# ==============================
# SIGNAL CHECKS (simple but safe)
# ==============================
def check_sell(df_h4: pd.DataFrame, df_h1: pd.DataFrame, regime: Regime, stop_sell_mode: bool) -> Tuple[bool, str]:
    if not regime.ok or stop_sell_mode:
        return False, "Regime not OK or STOP_SELL_MODE"

    c = df_h1["close"]
    price = float(c.iloc[-1])
    rsi_h1 = float(rsi(c, 14).iloc[-1])
    ema50_h1 = float(ema(c, 50).iloc[-1])

    low72 = float(df_h1["low"].tail(72).min())
    retrace = (price - low72) / low72 * 100 if low72 > 0 else 0

    # Sell setup: retrace + RSI high + at/above EMA50(H1)
    if retrace >= 4.0 and rsi_h1 > 62.0 and price >= ema50_h1:
        return True, f"Retrace={retrace:.1f}% RSI(H1)={rsi_h1:.1f} Price>=EMA50(H1)"
    return False, "No SELL setup"

def check_buy(df_h1: pd.DataFrame) -> Tuple[bool, str]:
    c = df_h1["close"]
    price = float(c.iloc[-1])
    rsi_h1 = float(rsi(c, 14).iloc[-1])
    high24 = float(df_h1["high"].tail(24).max())
    drop = (high24 - price) / high24 * 100 if high24 > 0 else 0

    if rsi_h1 < 38.0 or drop >= 6.0:
        return True, f"RSI(H1)={rsi_h1:.1f} DropFrom24hHigh={drop:.1f}%"
    return False, "No BUY setup"

# ==============================
# /CHECK HANDLER
# ==============================
def format_check_report(df_h4: pd.DataFrame, df_h1: pd.DataFrame, df_btc_h4: pd.DataFrame) -> str:
    regime = compute_regime(df_h4, df_btc_h4)
    stop_mode, stop_reason = compute_stop_sell_mode(df_h4)
    zones = compute_zones(df_h4, df_h1)

    rsi_h1 = float(rsi(df_h1["close"], 14).iloc[-1])
    rsi_h4 = float(rsi(df_h4["close"], 14).iloc[-1])

    sell_ok, sell_reason = check_sell(df_h4, df_h1, regime, stop_mode)
    buy_ok, buy_reason = check_buy(df_h1)

    # Compact sell zones text
    sz_lines = []
    for name, lo, hi in zones["sell_zones"]:
        sz_lines.append(f"- {name}: {lo:.2f} → {hi:.2f}")
    sell_zones_txt = "\n".join(sz_lines)

    bz_lo, bz_hi = zones["buy_zone"]

    d = regime.details
    report = (
        f"📌 /check {SYMBOL}\n"
        f"Time(UTC): H1={df_h1.index[-1].strftime('%Y-%m-%d %H:%M')} | H4={df_h4.index[-1].strftime('%Y-%m-%d %H:%M')}\n\n"
        f"Price: {zones['price']:.2f}\n\n"
        f"Regime(H4) score: {regime.score}/4\n"
        f"- EMA50<EMA200: {d['EMA50_lt_EMA200']}\n"
        f"- EMA200 slope down: {d['EMA200_slope_down']} (slope={regime.ema200_slope:.6f})\n"
        f"- Price<EMA200: {d['Price_lt_EMA200']}\n"
        f"- BTC confirm: {d['BTC_confirm']}\n\n"
        f"STOP_SELL_MODE: {stop_mode}" + (f" | {stop_reason}\n\n" if stop_mode else "\n\n")
        f"EMA(H1): EMA50={zones['ema50_h1']:.2f} | EMA200={zones['ema200_h1']:.2f}\n"
        f"EMA(H4): EMA50={zones['ema50_h4']:.2f} | EMA200={zones['ema200_h4']:.2f}\n"
        f"RSI: H1={rsi_h1:.1f} | H4={rsi_h4:.1f}\n"
        f"ATR(H1): {zones['atr_h1']:.2f}\n\n"
        f"SELL zones (vùng hồi để canh bán):\n{sell_zones_txt}\n\n"
        f"BUY zone (vùng hỗ trợ để canh mua lại):\n- Support band: {bz_lo:.2f} → {bz_hi:.2f} (support≈{zones['support']:.2f})\n\n"
        f"Signal now:\n"
        f"- SELL: {sell_ok} | {sell_reason}\n"
        f"- BUY:  {buy_ok} | {buy_reason}\n"
    )
    return report

def handle_telegram_commands(ex, state: dict) -> None:
    """
    Read new updates; if /check -> reply with computed report to the sender chat_id.
    Uses state['tg_update_offset'] for de-dup.
    """
    offset = state.get("tg_update_offset", None)
    resp = tg_get_updates(offset)
    if not resp.get("ok", False):
        return

    updates: List[dict] = resp.get("result", [])
    if not updates:
        return

    # Update offset to last_update_id + 1
    last_id = updates[-1].get("update_id")
    if last_id is not None:
        state["tg_update_offset"] = int(last_id) + 1
        save_state(state)

    for up in updates:
        msg = up.get("message") or up.get("edited_message")
        if not msg:
            continue

        text = (msg.get("text") or "").strip()
        chat = msg.get("chat") or {}
        chat_id = str(chat.get("id", "")).strip()
        if not chat_id:
            continue

        if text.startswith("/check"):
            try:
                # Fetch fresh data on demand
                df_h4 = fetch_df(ex, SYMBOL, TF_REGIME, limit=300)
                df_h1 = fetch_df(ex, SYMBOL, TF_ENTRY, limit=400)
                df_btc = fetch_df(ex, "BTC/USDT", TF_REGIME, limit=300)

                report = format_check_report(df_h4, df_h1, df_btc)
                tg_send(report, chat_id=chat_id)
            except Exception as e:
                tg_send(f"⚠️ /check error: {type(e).__name__}: {e}", chat_id=chat_id)

# ==============================
# MAIN LOOP
# ==============================
def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN")
    if not SYMBOL:
        raise RuntimeError("Missing SYMBOL")

    ex = make_exchange()
    state = load_state()

    # Startup message (push alert) – only if TELEGRAM_CHAT_ID is set
    if TELEGRAM_CHAT_ID:
        tg_send(f"Bot started for {SYMBOL}\nCommands: /check")

    last_signal_h1_ts = state.get("last_signal_h1_ts")
    stop_mode_prev = bool(state.get("stop_sell_mode", False))
    last_mode_h4_ts = state.get("last_mode_h4_ts")

    while True:
        try:
            # 1) Handle Telegram commands quickly (non-blocking)
            handle_telegram_commands(ex, state)

            # 2) Fetch market data for alert logic
            df_h4 = fetch_df(ex, SYMBOL, TF_REGIME, limit=300)
            df_h1 = fetch_df(ex, SYMBOL, TF_ENTRY, limit=400)
            df_btc = fetch_df(ex, "BTC/USDT", TF_REGIME, limit=300)

            regime = compute_regime(df_h4, df_btc)
            stop_mode, stop_reason = compute_stop_sell_mode(df_h4)

            # Mode change notifications (once per current H4 bar)
            h4_ts = df_h4.index[-1].isoformat()
            if stop_mode != stop_mode_prev and last_mode_h4_ts != h4_ts and TELEGRAM_CHAT_ID:
                if stop_mode:
                    tg_send(f"🟠 STOP SELL MODE\nReason: {stop_reason}")
                else:
                    tg_send("🟢 SELL MODE RESUMED")
                stop_mode_prev = stop_mode
                last_mode_h4_ts = h4_ts
                state["stop_sell_mode"] = stop_mode
                state["last_mode_h4_ts"] = h4_ts
                save_state(state)

            # Signals
            sell_ok, sell_reason = check_sell(df_h4, df_h1, regime, stop_mode)
            buy_ok, buy_reason = check_buy(df_h1)

            h1_ts = df_h1.index[-1].isoformat()

            # Dedup: only one signal per H1 candle
            if h1_ts != last_signal_h1_ts and TELEGRAM_CHAT_ID:
                price = float(df_h1["close"].iloc[-1])

                if sell_ok:
                    tg_send(
                        f"🚨 SELL SIGNAL {SYMBOL}\n"
                        f"Price: {price:.2f}\n"
                        f"Regime score: {regime.score}/4 | STOP_SELL_MODE={stop_mode}\n"
                        f"Reason: {sell_reason}"
                    )
                    last_signal_h1_ts = h1_ts

                elif buy_ok:
                    tg_send(
                        f"✅ BUY SIGNAL {SYMBOL}\n"
                        f"Price: {price:.2f}\n"
                        f"Reason: {buy_reason}"
                    )
                    last_signal_h1_ts = h1_ts

                state["last_signal_h1_ts"] = last_signal_h1_ts
                save_state(state)

            time.sleep(CHECK_INTERVAL_SEC)

        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}")
            time.sleep(15)

if __name__ == "__main__":
    main()