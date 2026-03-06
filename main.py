
# ===============================
# main.py — FastAPI + Supabase + RSI Bot (ETHUSDT & BTCUSDT)
# ===============================
import os
import json
import time
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import (
    FastAPI,
    Request,
    Form,
    APIRouter,
    Query,
)
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from supabase import create_client, Client


# ------------------------------------------------------------------
# 1) GLOBAL APP/ENV CONFIG
# ------------------------------------------------------------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")

def comma_format(value):
    try:
        return f"{float(value):,.0f}"
    except Exception:
        return value

templates.env.filters["comma"] = comma_format

logging.basicConfig(level=logging.INFO)

# Supabase ENV
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL và SUPABASE_KEY phải được thiết lập trong biến môi trường.")

if not SUPABASE_SERVICE_KEY:
    raise ValueError("SUPABASE_SERVICE_ROLE_KEY phải được thiết lập trong biến môi trường.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
supabase_admin: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


# ------------------------------------------------------------------
# 2) RSI BOT CONFIG
# ------------------------------------------------------------------
PUSHOVER_TOKEN = os.getenv("PUSHOVER_TOKEN", "")
PUSHOVER_USER = os.getenv("PUSHOVER_USER", "")
PUSHOVER_DEVICE = os.getenv("PUSHOVER_DEVICE", "")
RSI_SYMBOLS = [s.strip() for s in os.getenv("RSI_SYMBOLS", "ETHUSDT,BTCUSDT").split(",") if s.strip()] or ["ETHUSDT", "BTCUSDT"]
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
RSI_CHECK_MINUTES = int(os.getenv("RSI_CHECK_MINUTES", "5"))
RSI_TIMEFRAMES = {"1h": "1h", "4h": "4h", "1d": "1d"}

# ETH TRACKER CONFIG
ETH_TRACKER_SYMBOL = os.getenv("ETH_TRACKER_SYMBOL", "ETHUSDT")
ETH_TRACKER_INTERVAL = os.getenv("ETH_TRACKER_INTERVAL", "4h")
ETH_CYCLE_SIZE = float(os.getenv("ETH_CYCLE_SIZE", "40"))
ETH_BASE_BALANCE = float(os.getenv("ETH_BASE_BALANCE", "138"))

BIG_ORDER_THRESHOLD = 100_000
INTERVAL_MS_MAP = {
    "1m": 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "1h": 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}
TRACKER_INTERVAL = ETH_TRACKER_INTERVAL

ETH_SELL_ZONE_LOW = float(os.getenv("ETH_SELL_ZONE_LOW", "3650"))
ETH_SELL_ZONE_HIGH = float(os.getenv("ETH_SELL_ZONE_HIGH", "3700"))
ETH_BUY_ZONE_LOW = float(os.getenv("ETH_BUY_ZONE_LOW", "3350"))
ETH_BUY_ZONE_HIGH = float(os.getenv("ETH_BUY_ZONE_HIGH", "3450"))
ETH_RSI_SELL = float(os.getenv("ETH_RSI_SELL", "65"))
ETH_RSI_BUY = float(os.getenv("ETH_RSI_BUY", "40"))

MACD_FAST = int(os.getenv("ETH_MACD_FAST", "12"))
MACD_SLOW = int(os.getenv("ETH_MACD_SLOW", "26"))
MACD_SIGNAL = int(os.getenv("ETH_MACD_SIGNAL", "9"))

# State
_rsi_last_state: Dict[str, Dict[str, str]] = {sym: {tf: "unknown" for tf in RSI_TIMEFRAMES} for sym in RSI_SYMBOLS}
_rsi_last_values: Dict[str, Dict[str, Dict[str, float]]] = {}
_rsi_last_run: float = 0.0

# Router
_rsi_router = APIRouter()


# ------------------------------------------------------------------
# 3) INDICATOR FUNCTIONS
# ------------------------------------------------------------------
def _rsi_wilder(closes: List[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        raise ValueError("Not enough data to compute RSI")
    gains: List[float] = []
    losses: List[float] = []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def _rsi_fetch_klines(symbol: str, interval: str, limit: int = 200):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _compute_eth_zones_from_range(symbol: str, interval: str, lookback: int = 60):
    """
    Tính vùng BUY/SELL zone dựa trên high/low của N cây H4 gần nhất.
    - lookback: số nến dùng để tính (vd 60 nến H4 ≈ 10 ngày)
    Trả về: (sell_low, sell_high, buy_low, buy_high, recent_low, recent_high)
    """
    kl = _rsi_fetch_klines(symbol, interval, limit=lookback)
    if len(kl) < lookback:
        raise ValueError("Not enough klines for dynamic zone calc")

    highs = [float(k[2]) for k in kl]
    lows = [float(k[3]) for k in kl]

    recent_high = max(highs)
    recent_low = min(lows)
    price_range = recent_high - recent_low

    if price_range <= 0:
        raise ValueError("Invalid price range for ETH")

    zone_pct = 0.2

    buy_low = recent_low
    buy_high = recent_low + zone_pct * price_range

    sell_high = recent_high
    sell_low = recent_high - zone_pct * price_range

    return sell_low, sell_high, buy_low, buy_high, recent_low, recent_high


def _compute_ema_series(values: List[float], period: int) -> List[Optional[float]]:
    """
    Trả về list EMA cùng độ dài với values.
    Các phần tử đầu (chưa đủ period) sẽ là None.
    """
    if len(values) < period:
        raise ValueError(f"Not enough data for EMA({period})")

    ema_values: List[Optional[float]] = [None] * len(values)
    sma = sum(values[:period]) / period
    ema_values[period - 1] = sma

    k = 2 / (period + 1)
    ema_prev = sma
    for i in range(period, len(values)):
        ema = (values[i] - ema_prev) * k + ema_prev
        ema_values[i] = ema
        ema_prev = ema

    return ema_values


def _macd_latest(symbol: str, interval: str, fast: int = MACD_FAST, slow: int = MACD_SLOW, signal: int = MACD_SIGNAL):
    """
    Tính MACD (fast, slow, signal) cho symbol/interval.
    Trả về (macd_line, signal_line, hist) cho cây nến mới nhất.
    """
    limit = max(200, slow * 5)
    kl = _rsi_fetch_klines(symbol, interval, limit=limit)
    closes = [float(k[4]) for k in kl]

    if len(closes) < slow + signal + 5:
        raise ValueError("Not enough data to compute MACD")

    ema_fast = _compute_ema_series(closes, fast)
    ema_slow = _compute_ema_series(closes, slow)

    macd_series: List[float] = []
    for ef, es in zip(ema_fast, ema_slow):
        if ef is None or es is None:
            macd_series.append(0.0)
        else:
            macd_series.append(ef - es)

    signal_series = _compute_ema_series(macd_series, signal)

    macd_line = macd_series[-1]
    signal_line = signal_series[-1]
    if signal_line is None:
        raise ValueError("Signal line not ready")

    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _macd_latest_with_prev(
    symbol: str,
    interval: str,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
):
    """
    Tính MACD cho symbol/interval.
    Trả về (macd_line, signal_line, hist, prev_hist):
    - hist: histogram cây hiện tại
    - prev_hist: histogram cây liền trước
    """
    limit = max(200, slow * 5)
    kl = _rsi_fetch_klines(symbol, interval, limit=limit)
    closes = [float(k[4]) for k in kl]

    if len(closes) < slow + signal + 5:
        raise ValueError("Not enough data to compute MACD")

    ema_fast = _compute_ema_series(closes, fast)
    ema_slow = _compute_ema_series(closes, slow)

    macd_series: List[float] = []
    for ef, es in zip(ema_fast, ema_slow):
        if ef is None or es is None:
            macd_series.append(0.0)
        else:
            macd_series.append(ef - es)

    signal_series = _compute_ema_series(macd_series, signal)

    macd_line = macd_series[-1]
    signal_line = signal_series[-1]
    prev_signal_line = signal_series[-2]

    if signal_line is None or prev_signal_line is None:
        raise ValueError("Signal line not ready")

    hist = macd_line - signal_line
    prev_hist = macd_series[-2] - prev_signal_line

    return macd_line, signal_line, hist, prev_hist


def _rsi_latest(symbol: str, interval: str, period: int):
    kl = _rsi_fetch_klines(symbol, interval, limit=max(200, period * 5))
    closes = [float(k[4]) for k in kl]
    rsi = _rsi_wilder(closes, period=period)
    price = closes[-1]
    return price, rsi


def _pushover_notify(title: str, message: str):
    if not PUSHOVER_TOKEN or not PUSHOVER_USER:
        return
    data = {
        "token": PUSHOVER_TOKEN,
        "user": PUSHOVER_USER,
        "title": title,
        "message": message,
        "priority": 0,
        "sound": "cash",
    }
    if PUSHOVER_DEVICE:
        data["device"] = PUSHOVER_DEVICE
    try:
        requests.post("https://api.pushover.net/1/messages.json", data=data, timeout=15)
    except Exception:
        pass


def _fmt_dual(tf: str, condition: str, snapshot: Dict[str, Dict[str, float]]):
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S") + "Z"
    lines = [f"TF: {tf} | Cond: {condition} | RSI({RSI_PERIOD}) | {ts}"]
    ordered = sorted(snapshot.items(), key=lambda kv: (0 if kv[0].upper() == "ETHUSDT" else 1, kv[0]))
    for sym, v in ordered:
        if "price" in v and "rsi" in v:
            lines.append(f"{sym}: Price {v['price']:.2f} | RSI {v['rsi']:.2f}")
        else:
            lines.append(f"{sym}: error {v.get('error', 'unknown')}")
    return "\n".join(lines)


def _compute_rsi_series(closes: list[float], period: int) -> list[float]:
    """
    Tính RSI series classic từ list closes.
    Trả về list có cùng độ dài với closes (các giá trị đầu có thể bằng None -> thay bằng 50).
    """
    if len(closes) < period + 2:
        return [50.0] * len(closes)

    gains = []
    losses = []

    for i in range(1, len(closes)):
        change = closes[i] - closes[i - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    def ema(series, p):
        alpha = 2 / (p + 1)
        ema_vals = []
        prev = sum(series[:p]) / p
        ema_vals.append(prev)
        for v in series[p:]:
            prev = alpha * v + (1 - alpha) * prev
            ema_vals.append(prev)
        return ema_vals

    avg_gain = ema(gains, period)
    avg_loss = ema(losses, period)

    rsi = [50.0] * len(closes)
    offset = len(closes) - len(avg_gain)
    for i in range(len(avg_gain)):
        if avg_loss[i] == 0:
            rs = float('inf')
            r = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            r = 100 - (100 / (1 + rs))
        rsi[offset + i] = r

    return rsi


def _compute_macd_series(
    closes: list[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[list[float], list[float], list[float]]:
    """
    Tính MACD series cho 1 list closes.
    Trả về (macd_line[], signal_line[], hist[])
    """
    if len(closes) < slow + signal + 5:
        n = len(closes)
        return [0.0]*n, [0.0]*n, [0.0]*n

    ema_fast = _compute_ema_series(closes, fast)
    ema_slow = _compute_ema_series(closes, slow)

    macd_series: list[float] = []
    for ef, es in zip(ema_fast, ema_slow):
        if ef is None or es is None:
            macd_series.append(0.0)
        else:
            macd_series.append(ef - es)

    signal_series = _compute_ema_series(macd_series, signal)
    hist_series: list[float] = []
    for m, s in zip(macd_series, signal_series):
        if s is None:
            hist_series.append(0.0)
        else:
            hist_series.append(m - s)

    return macd_series, signal_series, hist_series


def _sma_series(values: list[float], period: int) -> list[float | None]:
    n = len(values)
    if n < period:
        return [None] * n
    out: list[float | None] = [None] * (period - 1)
    window_sum = sum(values[:period])
    out.append(window_sum / period)
    for i in range(period, n):
        window_sum += values[i] - values[i - period]
        out.append(window_sum / period)
    return out


def _bollinger_bands(values: list[float], period: int = 20, k: float = 2.0):
    """
    Trả về (middle[], upper[], lower[])
    middle = SMA(period)
    upper/lower = middle ± k * std
    """
    n = len(values)
    middle = _sma_series(values, period)
    upper: list[float | None] = [None] * n
    lower: list[float | None] = [None] * n

    if n < period:
        return middle, upper, lower

    import math

    for i in range(period - 1, n):
        window = values[i - period + 1 : i + 1]
        m = middle[i]
        if m is None:
            continue
        variance = sum((v - m) ** 2 for v in window) / period
        std = math.sqrt(variance)
        upper[i] = m + k * std
        lower[i] = m - k * std

    return middle, upper, lower


def _stochastic_oscillator(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 14,
) -> list[float | None]:
    """
    %K: 0..100
    """
    n = len(closes)
    if n < period:
        return [None] * n

    out: list[float | None] = [None] * n
    for i in range(period - 1, n):
        window_high = max(highs[i - period + 1 : i + 1])
        window_low = min(lows[i - period + 1 : i + 1])
        if window_high == window_low:
            out[i] = 50.0
        else:
            out[i] = (closes[i] - window_low) / (window_high - window_low) * 100.0
    return out


def _williams_r(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 14,
) -> list[float | None]:
    """
    Williams %R: -100 .. 0
    """
    n = len(closes)
    if n < period:
        return [None] * n

    out: list[float | None] = [None] * n
    for i in range(period - 1, n):
        window_high = max(highs[i - period + 1 : i + 1])
        window_low = min(lows[i - period + 1 : i + 1])
        if window_high == window_low:
            out[i] = -50.0
        else:
            out[i] = -100.0 * (window_high - closes[i]) / (window_high - window_low)
    return out


# ------------------------------------------------------------------
# 3b) D1 BIAS + CANDLE SIGNALS
# ------------------------------------------------------------------
def compute_d1_bias(symbol: str) -> tuple[bool, bool]:
    """Fetch D1 klines and return (d1_bullish, d1_bearish)."""
    klines_d1 = _rsi_fetch_klines(symbol, "1d", limit=200)
    closes_d1 = [float(k[4]) for k in klines_d1]
    ema12_d1 = _compute_ema_series(closes_d1, 12)
    ema26_d1 = _compute_ema_series(closes_d1, 26)
    sma50_d1 = _sma_series(closes_d1, 50)
    _, _, hist_d1 = _compute_macd_series(closes_d1, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    e12 = ema12_d1[-1]
    e26 = ema26_d1[-1]
    s50 = sma50_d1[-1]
    h   = hist_d1[-1]
    p   = closes_d1[-1]
    d1_bullish = (e12 is not None and e26 is not None and e12 > e26
                  and s50 is not None and p > s50 and h > -0.5)
    d1_bearish = (e12 is not None and e26 is not None and e12 < e26
                  and s50 is not None and p < s50 and h < 0.5)
    return d1_bullish, d1_bearish


def compute_candle_signals(
    closes, highs, lows, rsi, macd_hist, ema_fast, ema_slow,
    sma_50, bb_upper, bb_lower, stoch_k, williams_r,
    buy_zone_low, buy_zone_high, sell_zone_low, sell_zone_high,
    d1_bullish, d1_bearish, btc_rsi_h4, btc_macd_hist
) -> list[dict]:
    """
    Weighted scoring signal engine. Replaces the brittle 3-gate AND system.

    BUY score (max ~13):
      +3  price inside buy zone          (base, required — no zone = no signal)
      +1  RSI < 45
      +1  RSI < 35 (extra)
      +1  MACD hist improving (hist > prev)
      +1  MACD hist > 0
      +1  EMA12 > EMA26 (trend aligned)
      +1  Stoch %K < 30
      +1  Williams %R < -70
      +2  D1 macro bullish
      +1  price at/below BB lower band

    Fire BUY  when buy_score  >= 4 and not blocked.
    Fire SELL when sell_score >= 4 and not blocked.
    Strength: score >= 9 → 3, >= 6 → 2, else 1.
    """
    n = len(closes)
    results = []

    for i in range(n):
        price  = closes[i]
        rsi_i  = rsi[i]       if i < len(rsi)       else None
        macd_i = macd_hist[i] if i < len(macd_hist) else None
        macd_p = macd_hist[i - 1] if i > 0 and (i - 1) < len(macd_hist) else None
        ef     = ema_fast[i]  if i < len(ema_fast)  else None
        es     = ema_slow[i]  if i < len(ema_slow)  else None
        bb_u   = bb_upper[i]  if i < len(bb_upper)  else None
        bb_l   = bb_lower[i]  if i < len(bb_lower)  else None
        sk     = stoch_k[i]   if i < len(stoch_k)   else None
        wr_i   = williams_r[i] if i < len(williams_r) else None

        # ── BUY SCORE ─────────────────────────────────────────────
        buy_score = 0
        in_buy_zone = (
            buy_zone_low is not None and buy_zone_high is not None
            and buy_zone_low <= price <= buy_zone_high
        )
        if in_buy_zone:
            buy_score += 3
            if rsi_i is not None:
                if rsi_i < 45: buy_score += 1
                if rsi_i < 35: buy_score += 1
            if macd_i is not None and macd_p is not None:
                if macd_i > macd_p: buy_score += 1   # momentum improving
                if macd_i > 0:      buy_score += 1   # positive territory
            if ef is not None and es is not None and ef > es:
                buy_score += 1                         # EMA trend aligned
            if sk  is not None and sk  < 30:  buy_score += 1
            if wr_i is not None and wr_i < -70: buy_score += 1
            if d1_bullish:                    buy_score += 2
            if bb_l is not None and price <= bb_l: buy_score += 1

        # ── SELL SCORE ────────────────────────────────────────────
        sell_score = 0
        in_sell_zone = (
            sell_zone_low is not None and sell_zone_high is not None
            and sell_zone_low <= price <= sell_zone_high
        )
        if in_sell_zone:
            sell_score += 3
            if rsi_i is not None:
                if rsi_i > 55: sell_score += 1
                if rsi_i > 65: sell_score += 1
            if macd_i is not None and macd_p is not None:
                if macd_i < macd_p: sell_score += 1   # momentum fading
                if macd_i < 0:      sell_score += 1   # negative territory
            if ef is not None and es is not None and ef < es:
                sell_score += 1                         # EMA trend aligned
            if sk   is not None and sk   > 70:  sell_score += 1
            if wr_i is not None and wr_i > -30: sell_score += 1
            if d1_bearish:                      sell_score += 2
            if bb_u is not None and price >= bb_u: sell_score += 1

        # ── DANGER ZONE (display tint + buy blocker) ─────────────
        dz_flags = []
        if d1_bearish:
            dz_flags.append("D1Bear")
        if bb_u is not None and rsi_i is not None and price >= bb_u and rsi_i > 72:
            dz_flags.append("OBought")
        if (btc_rsi_h4 is not None and btc_macd_hist is not None
                and btc_rsi_h4 < 35 and btc_macd_hist < 0):
            dz_flags.append("BTCweak")
        is_danger_zone = len(dz_flags) > 0

        # ── SIGNAL BLOCKING ───────────────────────────────────────
        # Suppress buys when macro is bearish or BTC is crashing
        buy_blocked  = is_danger_zone
        # Suppress sells when BTC is strongly bullish (don't sell into strong uptrend)
        btc_bull = (btc_rsi_h4 is not None and btc_rsi_h4 > 65
                    and btc_macd_hist is not None and btc_macd_hist > 0)
        sell_blocked = d1_bullish or btc_bull

        # ── FINAL SIGNALS ─────────────────────────────────────────
        is_buy_signal  = buy_score  >= 4 and not buy_blocked
        is_sell_signal = sell_score >= 4 and not sell_blocked

        buy_strength  = 3 if buy_score  >= 9 else (2 if buy_score  >= 6 else 1)
        sell_strength = 3 if sell_score >= 9 else (2 if sell_score >= 6 else 1)
        signal_strength = (
            buy_strength  if is_buy_signal  else
            sell_strength if is_sell_signal else 0
        )

        # ── ZONE LABEL ────────────────────────────────────────────
        if in_buy_zone:
            zone = "buy"
        elif in_sell_zone:
            zone = "sell"
        else:
            zone = "neutral"

        # ── DETAIL STRING (tooltip) ───────────────────────────────
        parts = []
        if is_buy_signal:
            parts.append(f"BUY str={buy_strength} score={buy_score}")
        if is_sell_signal:
            parts.append(f"SELL str={sell_strength} score={sell_score}")
        if dz_flags:
            parts.append("DANGER:" + ",".join(dz_flags))
        if not parts:
            parts.append(f"zone={zone} b={buy_score} s={sell_score}")
        signal_detail = " | ".join(parts)

        results.append({
            "is_buy_signal":   bool(is_buy_signal),
            "is_sell_signal":  bool(is_sell_signal),
            "is_danger_zone":  bool(is_danger_zone),
            "signal_strength": signal_strength,
            "buy_score":       buy_score,
            "sell_score":      sell_score,
            "zone":            zone,
            "signal_detail":   signal_detail,
        })

    return results


# ------------------------------------------------------------------
# 4) RSI BOT LOGIC
# ------------------------------------------------------------------
def _rsi_check_once():
    global _rsi_last_state, _rsi_last_values, _rsi_last_run
    snap_all: Dict[str, Dict[str, Dict[str, float]]] = {}

    for tf, interval in RSI_TIMEFRAMES.items():
        tf_snap: Dict[str, Dict[str, float]] = {}

        for sym in RSI_SYMBOLS:
            try:
                price, rsi = _rsi_latest(sym, interval, RSI_PERIOD)
                tf_snap[sym] = {"price": price, "rsi": rsi}
            except Exception as e:
                tf_snap[sym] = {"error": str(e)}

        for sym in RSI_SYMBOLS:
            v = tf_snap.get(sym, {})
            rsi = v.get("rsi")
            if rsi is None:
                continue
            prev = _rsi_last_state.get(sym, {}).get(tf, "unknown")
            if rsi < 30 and prev != "oversold":
                _pushover_notify(f"RSI Oversold {tf} — {sym}", _fmt_dual(tf, "<30", tf_snap))
                _rsi_last_state[sym][tf] = "oversold"
            elif rsi > 70 and prev != "overbought":
                _pushover_notify(f"RSI Overbought {tf} — {sym}", _fmt_dual(tf, ">70", tf_snap))
                _rsi_last_state[sym][tf] = "overbought"
            elif 30 <= rsi <= 70 and prev != "normal":
                _rsi_last_state[sym][tf] = "normal"

        snap_all[tf] = tf_snap

    _rsi_last_values = snap_all
    _rsi_last_run = time.time()
    return snap_all


def _eth_decide_action(
    price: float,
    rsi_h4: float,
    macd_hist: float,
    prev_macd_hist: float,
    zones: tuple,
    btc_rsi_h4: float,
    btc_macd_hist: float,
    btc_prev_macd_hist: float,
) -> Dict[str, str]:
    """
    Quyết định BUY/SELL/HOLD với:
    - zones: (sell_low, sell_high, buy_low, buy_high, recent_low, recent_high)
    - BTC filter để tránh bán ngược trend.
    """
    sell_low, sell_high, buy_low, buy_high, recent_low, recent_high = zones

    reasons: List[str] = []
    reasons.append(
        f"Dynamic zones: BUY[{buy_low:.1f}-{buy_high:.1f}] "
        f"SELL[{sell_low:.1f}-{sell_high:.1f}] "
        f"(range {recent_low:.1f}-{recent_high:.1f})"
    )

    action = "HOLD"

    macd_weakening = (
        macd_hist > 0
        and prev_macd_hist is not None
        and macd_hist < prev_macd_hist
    )

    if (
        sell_low <= price <= sell_high
        and rsi_h4 >= ETH_RSI_SELL
        and macd_weakening
    ):
        action = "SELL"
        reasons.append(
            f"Price {price:.1f} in SELL zone & RSI_H4 {rsi_h4:.1f} >= {ETH_RSI_SELL}"
        )
        reasons.append(
            f"MACD hist weakening: current {macd_hist:.4f} < prev {prev_macd_hist:.4f}"
        )

    elif buy_low <= price <= buy_high and rsi_h4 <= ETH_RSI_BUY:
        action = "BUY"
        reasons.append(
            f"Price {price:.1f} in BUY zone & RSI_H4 {rsi_h4:.1f} <= {ETH_RSI_BUY}"
        )

    else:
        reasons.append("No buy/sell condition matched (HOLD).")

    btc_bull_rsi = btc_rsi_h4 >= 65
    btc_macd_stronger = (
        btc_macd_hist > 0
        and btc_prev_macd_hist is not None
        and btc_macd_hist >= btc_prev_macd_hist
    )

    if action == "SELL" and (btc_bull_rsi or btc_macd_stronger):
        reasons.append(
            f"Cancel SELL: BTC still bullish (RSI_H4={btc_rsi_h4:.1f}, "
            f"MACD hist {btc_macd_hist:.4f} >= prev {btc_prev_macd_hist:.4f})"
        )
        action = "HOLD"

    if abs(macd_hist) < 0.5:
        reasons.append("MACD hist ~0 → momentum weak / sideway.")
    elif macd_hist > 0:
        reasons.append("MACD hist > 0 → bullish momentum.")
    else:
        reasons.append("MACD hist < 0 → bearish momentum.")

    return {
        "action": action,
        "reason": " | ".join(reasons),
    }


def run_symbol_tracker_once(symbol: str, send_notify: bool = False) -> Dict[str, Any]:
    """
    Tracker chung cho mọi symbol:
    - Lấy price + RSI H4 từ Binance
    - MACD + prev hist
    - Dynamic zone (dùng logic _compute_eth_zones_from_range)
    - BTC filter (BTC RSI + MACD hist + prev hist)
    - Quyết định action bằng _eth_decide_action
    - Không ghi DB, chỉ trả payload (+ optional Pushover).
    """
    symbol = symbol.upper()

    interval = TRACKER_INTERVAL

    price, rsi_h4 = _rsi_latest(symbol, interval, RSI_PERIOD)

    macd_line, macd_signal, macd_hist, prev_macd_hist = _macd_latest_with_prev(
        symbol,
        interval,
    )

    btc_price, btc_rsi_h4 = _rsi_latest("BTCUSDT", interval, RSI_PERIOD)

    _, _, btc_macd_hist, btc_prev_macd_hist = _macd_latest_with_prev(
        "BTCUSDT",
        interval,
    )

    zones = _compute_eth_zones_from_range(symbol, interval, lookback=60)
    sell_low, sell_high, buy_low, buy_high, recent_low, recent_high = zones

    decision = _eth_decide_action(
        price=price,
        rsi_h4=rsi_h4,
        macd_hist=macd_hist,
        prev_macd_hist=prev_macd_hist,
        zones=zones,
        btc_rsi_h4=btc_rsi_h4,
        btc_macd_hist=btc_macd_hist,
        btc_prev_macd_hist=btc_prev_macd_hist,
    )
    action = decision["action"]
    reason = decision["reason"]

    now_utc = datetime.utcnow().isoformat() + "Z"

    payload: Dict[str, Any] = {
        "symbol": symbol,
        "timeframe": interval,
        "now_utc": now_utc,
        "price": price,
        "rsi_h4": rsi_h4,
        "macd": macd_line,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "action": action,
        "reason": reason,
        "zones": {
            "sell_low": sell_low,
            "sell_high": sell_high,
            "buy_low": buy_low,
            "buy_high": buy_high,
            "recent_low": recent_low,
            "recent_high": recent_high,
        },
        "btc": {
            "price": btc_price,
            "rsi_h4": btc_rsi_h4,
            "macd_hist": btc_macd_hist,
            "prev_macd_hist": btc_prev_macd_hist,
        },
    }

    if send_notify and action != "HOLD":
        try:
            title = f"[{action}] {symbol}"
            msg_lines = [
                f"Price: {price}",
                f"Reason: {reason}",
                f"RSI H4: {rsi_h4:.2f}",
                f"MACD: {macd_line:.4f} | Signal: {macd_signal:.4f} | Hist: {macd_hist:.4f}",
                f"BTC RSI H4: {btc_rsi_h4:.1f}, BTC hist: {btc_macd_hist:.4f}",
                f"Time (UTC): {now_utc}",
            ]
            _pushover_notify(title, "\n".join(msg_lines))
        except Exception as e:
            logging.error(f"[SYMBOL_TRACKER_NOTIFY] Error: {e}")

    return payload


def symbols_tracker_job():
    """
    Job chạy mỗi 10 phút:
    - Lấy danh sách symbol is_active = true trong bot_subscriptions
    - Mỗi symbol → run_symbol_tracker_once(send_notify=True)
    """
    try:
        resp = (
            supabase_admin.table("bot_subscriptions")
            .select("symbol")
            .eq("is_active", True)
            .execute()
        )
        rows = resp.data or []
    except Exception as e:
        logging.error(f"[SYMBOL_TRACKER_JOB] Error fetch subscriptions: {e}")
        return

    for row in rows:
        symbol = (row.get("symbol") or "").upper()
        if not symbol:
            continue
        try:
            payload = run_symbol_tracker_once(symbol, send_notify=True)
            logging.info(
                f"[SYMBOL_TRACKER_JOB] {symbol}: action={payload['action']} price={payload['price']}"
            )
        except Exception as e:
            logging.error(f"[SYMBOL_TRACKER_JOB] {symbol}: error {e}")


def init_inline_rsi_dual(app_: FastAPI, scheduler: Optional[BackgroundScheduler] = None):
    app_.include_router(_rsi_router, prefix="/bots", tags=["bots"])
    if scheduler is not None:
        try:
            scheduler.add_job(
                symbols_tracker_job,
                "interval",
                minutes=10,
                id="symbols_tracker_job",
                replace_existing=True,
                next_run_time=datetime.utcnow(),
            )
        except Exception:
            scheduler.add_job(
                symbols_tracker_job,
                "interval",
                minutes=10,
                id="symbols_tracker_job",
                replace_existing=True,
            )
    else:
        import threading

        def _loop():
            while True:
                try:
                    _rsi_check_once()
                except Exception:
                    pass
                time.sleep(RSI_CHECK_MINUTES * 60)

        threading.Thread(target=_loop, daemon=True).start()


# ------------------------------------------------------------------
# 5) BOT ENDPOINTS
# ------------------------------------------------------------------
@_rsi_router.post("/subscribe")
async def subscribe_symbol(symbol: str = Form(...)):
    """
    SUBSCRIBE 1 symbol vào danh sách theo dõi.
    """
    symbol = symbol.upper()
    try:
        supabase_admin.table("bot_subscriptions") \
            .upsert(
                {"symbol": symbol, "is_active": True},
                on_conflict="symbol",
            ) \
            .execute()
    except Exception as e:
        logging.error(f"[SUBSCRIBE] Error subscribe {symbol}: {e}")

    return RedirectResponse(url=f"/bots/{symbol}", status_code=303)


@_rsi_router.post("/unsubscribe")
async def unsubscribe_symbol(symbol: str = Form(...)):
    """
    UNSUBSCRIBE 1 symbol khỏi danh sách theo dõi.
    """
    symbol = symbol.upper()
    try:
        supabase_admin.table("bot_subscriptions") \
            .update({"is_active": False}) \
            .eq("symbol", symbol) \
            .execute()
    except Exception as e:
        logging.error(f"[UNSUBSCRIBE] Error unsubscribe {symbol}: {e}")

    return RedirectResponse(url=f"/bots/{symbol}", status_code=303)


@_rsi_router.get("/{symbol}", response_class=HTMLResponse)
async def symbol_dashboard(request: Request, symbol: str, tf: str = Query("4h")):
    """
    Dashboard theo dõi bất kỳ symbol nào (BTCUSDT, ETHUSDT, BNBUSDT...):
    - Giá, RSI, %change 24h
    - Buy/Sell zone (dynamic)
    - BUY/SELL signals (server-side: multi-gate with D1 bias)
    - tracker_action/server (BUY/SELL/HOLD) dùng cùng logic với bot (ethtracker)
    - SUBSCRIBE/UNSUBSCRIBE symbol này.
    - tf: user-selected timeframe (1h, 4h, 1d)
    """
    symbol = symbol.upper()

    # Validate and map tf
    valid_tfs = {"1h": "1h", "4h": "4h", "1d": "1d"}
    if tf not in valid_tfs:
        tf = "4h"
    interval = valid_tfs[tf]

    try:
        klines = _rsi_fetch_klines(symbol, interval, limit=200)
    except Exception as e:
        logging.error(f"[SYMBOL DASH] Error fetching klines for {symbol}: {e}")
        klines = []

    labels: List[str] = []
    closes: List[float] = []
    highs: List[float] = []
    lows: List[float] = []

    for k in klines:
        try:
            open_time_ms = int(k[0])
            dt = datetime.utcfromtimestamp(open_time_ms / 1000.0)
            labels.append(dt.strftime("%Y-%m-%d %H:%M"))

            highs.append(float(k[2]))
            lows.append(float(k[3]))
            closes.append(float(k[4]))
        except Exception as e:
            logging.warning(f"[SYMBOL DASH] Bad kline row for {symbol}: {e}")
            continue

    if not closes:
        context = {
            "request": request,
            "symbol": symbol,
            "tf": tf,
            "rows_json_str": "[]",
            "last_price": None,
            "last_rsi": None,
            "change_24h": None,
            "buy_low": None,
            "buy_high": None,
            "sell_low": None,
            "sell_high": None,
            "is_subscribed": False,
            "tracker_action": "HOLD",
            "tracker_reason": "No data",
            "d1_bullish": False,
            "d1_bearish": False,
        }
        return templates.TemplateResponse("symbol_dashboard.html", context)

    rsi_values = _compute_rsi_series(closes, RSI_PERIOD)
    macd_line_series, macd_signal_series, macd_hist_values = _compute_macd_series(
        closes, MACD_FAST, MACD_SLOW, MACD_SIGNAL
    )
    ema_fast = _compute_ema_series(closes, 12)
    ema_slow = _compute_ema_series(closes, 26)
    sma_50 = _sma_series(closes, 50)

    bb_middle, bb_upper, bb_lower = _bollinger_bands(closes, period=20, k=2.0)
    stoch_k = _stochastic_oscillator(highs, lows, closes, period=14)
    williams_r_vals = _williams_r(highs, lows, closes, period=14)

    buy_low = buy_high = sell_low = sell_high = recent_low = recent_high = None
    try:
        zones = _compute_eth_zones_from_range(symbol, interval, lookback=60)
        sell_low, sell_high, buy_low, buy_high, recent_low, recent_high = zones
    except Exception as e:
        logging.error(f"[SYMBOL DASH] Error computing zones for {symbol}: {e}")

    n = len(closes)
    min_len = min(
        n,
        len(labels),
        len(rsi_values),
        len(macd_hist_values),
        len(ema_fast),
        len(ema_slow),
        len(bb_upper),
        len(bb_lower),
        len(stoch_k),
        len(williams_r_vals),
    )

    labels = labels[-min_len:]
    closes_trimmed = closes[-min_len:]
    highs_trimmed = highs[-min_len:]
    lows_trimmed = lows[-min_len:]
    rsi_values = rsi_values[-min_len:]
    macd_hist_values = macd_hist_values[-min_len:]
    ema_fast = ema_fast[-min_len:]
    ema_slow = ema_slow[-min_len:]
    sma_50 = sma_50[-min_len:]
    bb_middle = bb_middle[-min_len:]
    bb_upper = bb_upper[-min_len:]
    bb_lower = bb_lower[-min_len:]
    stoch_k = stoch_k[-min_len:]
    williams_r_vals = williams_r_vals[-min_len:]

    rows_json: List[Dict[str, Any]] = []
    for i in range(min_len):
        rows_json.append(
            {
                "time_str": labels[i],
                "price": closes_trimmed[i],
                "high": highs_trimmed[i],
                "low": lows_trimmed[i],
                "rsi_h4": rsi_values[i],
                "macd_hist": macd_hist_values[i],
                "ema_fast": ema_fast[i],
                "ema_slow": ema_slow[i],
                "sma_50": sma_50[i],
                "bb_middle": bb_middle[i],
                "bb_upper": bb_upper[i],
                "bb_lower": bb_lower[i],
                "stoch_k": stoch_k[i],
                "wr": williams_r_vals[i],
            }
        )

    # --- D1 Bias ---
    d1_bullish = False
    d1_bearish = False
    try:
        d1_bullish, d1_bearish = compute_d1_bias(symbol)
    except Exception as e:
        logging.error(f"[SYMBOL DASH] Error computing D1 bias for {symbol}: {e}")

    # --- BTC context for signal computation ---
    btc_rsi_h4 = None
    btc_macd_hist_val = None
    try:
        _, btc_rsi_h4 = _rsi_latest("BTCUSDT", interval, RSI_PERIOD)
        _, _, btc_macd_hist_val, _ = _macd_latest_with_prev("BTCUSDT", interval)
    except Exception as e:
        logging.error(f"[SYMBOL DASH] Error fetching BTC context: {e}")

    # --- Compute per-candle signals ---
    signals = compute_candle_signals(
        closes=closes_trimmed,
        highs=highs_trimmed,
        lows=lows_trimmed,
        rsi=rsi_values,
        macd_hist=macd_hist_values,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        sma_50=sma_50,
        bb_upper=bb_upper,
        bb_lower=bb_lower,
        stoch_k=stoch_k,
        williams_r=williams_r_vals,
        buy_zone_low=buy_low,
        buy_zone_high=buy_high,
        sell_zone_low=sell_low,
        sell_zone_high=sell_high,
        d1_bullish=d1_bullish,
        d1_bearish=d1_bearish,
        btc_rsi_h4=btc_rsi_h4,
        btc_macd_hist=btc_macd_hist_val,
    )

    # Merge signal data into rows_json
    for i, row in enumerate(rows_json):
        row.update(signals[i])

    last_price = closes_trimmed[-1]
    last_rsi = rsi_values[-1] if rsi_values else None

    change_24h = None
    try:
        if len(closes_trimmed) >= 7:
            ref = closes_trimmed[-7]
            if ref != 0:
                change_24h = (last_price - ref) / ref * 100.0
    except Exception:
        change_24h = None

    # run_symbol_tracker_once always uses TRACKER_INTERVAL (4h) for server-side action
    tracker_action = "HOLD"
    tracker_reason = ""
    try:
        payload = run_symbol_tracker_once(symbol, send_notify=False)
        tracker_action = payload.get("action", "HOLD")
        tracker_reason = payload.get("reason", "")
    except Exception as e:
        logging.error(f"[SYMBOL DASH] Error run_symbol_tracker_once for {symbol}: {e}")

    is_subscribed = False
    try:
        resp = (
            supabase_admin.table("bot_subscriptions")
            .select("is_active")
            .eq("symbol", symbol)
            .limit(1)
            .execute()
        )
        db_rows = resp.data or []
        if db_rows and db_rows[0].get("is_active"):
            is_subscribed = True
    except Exception as e:
        logging.error(f"[SYMBOL DASH] Error check subscription for {symbol}: {e}")

    rows_json_str = json.dumps(rows_json)

    context = {
        "request": request,
        "symbol": symbol,
        "tf": tf,
        "rows_json_str": rows_json_str,
        "last_price": last_price,
        "last_rsi": last_rsi,
        "change_24h": change_24h,
        "buy_low": buy_low,
        "buy_high": buy_high,
        "sell_low": sell_low,
        "sell_high": sell_high,
        "is_subscribed": is_subscribed,
        "tracker_action": tracker_action,
        "tracker_reason": tracker_reason,
        "d1_bullish": d1_bullish,
        "d1_bearish": d1_bearish,
    }

    return templates.TemplateResponse("symbol_dashboard.html", context)


# ------------------------------------------------------------------
# 6) SCHEDULER INIT
# ------------------------------------------------------------------
scheduler = BackgroundScheduler()
init_inline_rsi_dual(app, scheduler)
scheduler.start()


# ------------------------------------------------------------------
# 7) ENTRY POINT
# ------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=True,
        workers=1,
    )
