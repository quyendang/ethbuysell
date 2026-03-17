
# ===============================
# main.py — FastAPI + Supabase + RSI Bot (ETHUSDT & BTCUSDT)
# ===============================
import asyncio
import os
import json
import time
import threading
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
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_IDS = [cid.strip() for cid in os.getenv("TELEGRAM_CHAT_IDS", "").split(",") if cid.strip()]
RSI_SYMBOLS = [s.strip() for s in os.getenv("RSI_SYMBOLS", "ETHUSDT,BTCUSDT").split(",") if s.strip()] or ["ETHUSDT", "BTCUSDT"]
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
RSI_CHECK_MINUTES = int(os.getenv("RSI_CHECK_MINUTES", "5"))
RSI_TIMEFRAMES = {"1h": "1h", "4h": "4h", "1d": "1d"}
# Minimum minutes between repeated notifications for the same symbol+action.
# Default: 240 min (4h) — matches the default tracking interval.
NOTIFY_COOLDOWN_MINUTES = int(os.getenv("NOTIFY_COOLDOWN_MINUTES", "120"))

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

# OpenRouter AI
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "google/gemini-flash-1.5")

# State
_rsi_last_state: Dict[str, Dict[str, str]] = {sym: {tf: "unknown" for tf in RSI_TIMEFRAMES} for sym in RSI_SYMBOLS}
_rsi_last_values: Dict[str, Dict[str, Dict[str, float]]] = {}
_rsi_last_run: float = 0.0
# Notification cooldown: tracks last sent timestamp per "SYMBOL_ACTION" key
_notify_last_sent: Dict[str, float] = {}

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


# ── Klines TTL cache (5 min) — eliminates duplicate Binance fetches ──────────
_klines_cache: Dict[str, tuple] = {}   # key=(symbol_interval) -> (ts, data[250])
_klines_cache_lock = threading.Lock()
KLINES_CACHE_TTL = 300  # seconds


def _rsi_fetch_klines(symbol: str, interval: str, limit: int = 200):
    """Fetch klines from Binance with a 5-minute in-memory cache.

    Always fetches 250 candles internally and slices to `limit`, so all
    calls for the same (symbol, interval) share the same cache entry
    regardless of the requested limit.
    """
    cache_key = f"{symbol}_{interval}"
    now = time.time()
    with _klines_cache_lock:
        entry = _klines_cache.get(cache_key)
        if entry and now - entry[0] < KLINES_CACHE_TTL:
            data = entry[1]
            return data[-limit:] if len(data) >= limit else data

    url = "https://api.binance.com/api/v3/klines"
    fetch_limit = max(limit, 250)
    params = {"symbol": symbol, "interval": interval, "limit": fetch_limit}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    with _klines_cache_lock:
        _klines_cache[cache_key] = (time.time(), data)

    return data[-limit:] if len(data) >= limit else data


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


def _can_notify(symbol: str, action: str) -> bool:
    """Return True if enough time has passed since the last notification for this symbol+action."""
    key = f"{symbol}_{action}"
    last = _notify_last_sent.get(key, 0.0)
    return (time.time() - last) >= NOTIFY_COOLDOWN_MINUTES * 60


def _mark_notified(symbol: str, action: str) -> None:
    """Record that a notification was just sent for symbol+action."""
    _notify_last_sent[f"{symbol}_{action}"] = time.time()
    # When BUY fires, reset SELL cooldown (and vice versa) so a direction flip always notifies.
    opposite = "SELL" if action == "BUY" else "BUY"
    _notify_last_sent.pop(f"{symbol}_{opposite}", None)


def _telegram_notify(title: str, message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_IDS:
        return
    text = f"<b>{title}</b>\n{message}"
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    for chat_id in TELEGRAM_CHAT_IDS:
        try:
            requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=15)
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
    Uses Wilder's smoothing (same as _rsi_wilder) for consistency.
    """
    if len(closes) < period + 2:
        return [50.0] * len(closes)
    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rsi = [50.0] * len(closes)
    # seed
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rsi[period] = 100 - (100 / (1 + avg_gain / avg_loss))
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        idx = i + 1
        if avg_loss == 0:
            rsi[idx] = 100.0
        else:
            rsi[idx] = 100 - (100 / (1 + avg_gain / avg_loss))
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


def _compute_atr_series(
    highs: list[float], lows: list[float], closes: list[float], period: int = 14
) -> list[float]:
    """ATR series using Wilder smoothing. Returns list same length as closes, None for early bars."""
    n = len(closes)
    if n < 2:
        return [None] * n
    tr_series = [None]
    for i in range(1, n):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        tr_series.append(tr)
    atr_series = [None] * n
    if n <= period:
        return atr_series
    seed = sum(tr_series[1:period+1]) / period
    atr_series[period] = seed
    prev = seed
    for i in range(period + 1, n):
        prev = (prev * (period - 1) + tr_series[i]) / period
        atr_series[i] = prev
    return atr_series


def _compute_adx(
    highs: list[float], lows: list[float], closes: list[float], period: int = 14
) -> dict:
    """Compute latest ADX, +DI, -DI for trend strength detection."""
    n = len(closes)
    if n < period * 2 + 5:
        return {"adx": 20.0, "di_plus": 20.0, "di_minus": 20.0, "trending": False, "strong": False}
    tr_list, dmp_list, dmm_list = [0.0], [0.0], [0.0]
    for i in range(1, n):
        tr = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
        tr_list.append(tr)
        up, down = highs[i]-highs[i-1], lows[i-1]-lows[i]
        dmp_list.append(up if up > down and up > 0 else 0.0)
        dmm_list.append(down if down > up and down > 0 else 0.0)
    def _wilder(s, p):
        r = [sum(s[:p])]
        for v in s[p:]:
            r.append(r[-1] - r[-1]/p + v)
        return r
    atr_s = _wilder(tr_list, period)
    dmp_s = _wilder(dmp_list, period)
    dmm_s = _wilder(dmm_list, period)
    di_p = [100*d/a if a > 0 else 0 for d, a in zip(dmp_s, atr_s)]
    di_m = [100*d/a if a > 0 else 0 for d, a in zip(dmm_s, atr_s)]
    dx_s = [100*abs(p-m)/(p+m) if (p+m) > 0 else 0 for p, m in zip(di_p, di_m)]
    if len(dx_s) < period:
        return {"adx": 20.0, "di_plus": 20.0, "di_minus": 20.0, "trending": False, "strong": False}
    adx_s = _wilder(dx_s, period)
    adx = adx_s[-1]
    return {
        "adx": round(adx, 2),
        "di_plus": round(di_p[-1], 2),
        "di_minus": round(di_m[-1], 2),
        "trending": adx > 22.0,
        "strong": adx > 30.0,
    }


def compute_dynamic_rsi_thresholds(
    rsi_series: list[float], lookback: int = 100,
) -> tuple[float, float]:
    """
    Compute adaptive RSI oversold/overbought thresholds from recent RSI distribution.
    Returns (oversold_threshold, overbought_threshold).
    """
    recent = [r for r in rsi_series[-lookback:] if r is not None and r > 0]
    if len(recent) < 20:
        return 40.0, 60.0
    s = sorted(recent)
    n = len(s)
    def pct(data, p):
        idx = (len(data)-1) * p / 100.0
        lo, hi = int(idx), min(int(idx)+1, len(data)-1)
        return data[lo] + (data[hi]-data[lo]) * (idx-lo)
    oversold = max(25.0, min(50.0, pct(s, 20)))
    overbought = max(50.0, min(78.0, pct(s, 80)))
    if overbought - oversold < 10.0:
        mid = (oversold + overbought) / 2
        oversold, overbought = mid - 5.0, mid + 5.0
    return round(oversold, 1), round(overbought, 1)


def compute_obv_signals(
    closes: list[float], volumes: list[float], lookback: int = 8
) -> dict:
    """
    Compute OBV trend and VWAP position for volume-based scoring.
    Returns buy_vol_score (0-3) and sell_vol_score (0-3).
    """
    n = len(closes)
    if n < 2 or not volumes:
        return {"buy_vol_score": 0, "sell_vol_score": 0, "obv_trend": "flat", "price_vs_vwap": "near"}
    obv = [0.0]
    for i in range(1, n):
        if closes[i] > closes[i-1]:
            obv.append(obv[-1] + volumes[i])
        elif closes[i] < closes[i-1]:
            obv.append(obv[-1] - volumes[i])
        else:
            obv.append(obv[-1])
    w = min(lookback, n)
    half = max(1, w // 2)
    obv_start = sum(obv[-w:-half]) / half if half < w else obv[-w]
    obv_end = sum(obv[-half:]) / half
    change = (obv_end - obv_start) / (abs(obv_start) + 1e-9)
    obv_trend = "up" if change > 0.005 else ("down" if change < -0.005 else "flat")
    vw = min(20, n)
    avg_vol = sum(volumes[-vw:]) / vw
    curr_vol = volumes[-1]
    vol_spike = curr_vol > avg_vol * 1.5
    candle_up = closes[-1] > closes[-2]
    candle_dn = closes[-1] < closes[-2]
    price_change = abs(closes[-1] - closes[-2]) / closes[-2] if closes[-2] > 0 else 0
    obv_bull_div = closes[-1] < closes[max(0, n-10)] and obv[-1] > obv[max(0, n-10)]
    obv_bear_div = closes[-1] > closes[max(0, n-10)] and obv[-1] < obv[max(0, n-10)]
    buy_score = 0
    sell_score = 0
    if obv_trend == "up": buy_score += 1
    if obv_bull_div: buy_score += 2
    if vol_spike and candle_up and price_change > 0.003: buy_score += 1
    if obv_trend == "down": sell_score += 1
    if obv_bear_div: sell_score += 2
    if vol_spike and candle_dn and price_change > 0.003: sell_score += 1
    return {
        "buy_vol_score": min(buy_score, 3),
        "sell_vol_score": min(sell_score, 3),
        "obv_trend": obv_trend,
        "vol_spike": vol_spike,
    }


def _fetch_fear_greed() -> dict:
    """Fetch Fear & Greed Index from alternative.me. Cached by caching the result in module-level dict."""
    try:
        resp = requests.get(
            "https://api.alternative.me/fng/",
            params={"limit": 1, "format": "json"},
            timeout=8,
        )
        resp.raise_for_status()
        entry = resp.json()["data"][0]
        value = int(entry["value"])
        return {
            "value": value,
            "buy_adj": 2 if value <= 20 else (1 if value <= 35 else 0),
            "sell_adj": 2 if value >= 80 else (1 if value >= 65 else 0),
        }
    except Exception:
        return {"value": 50, "buy_adj": 0, "sell_adj": 0}


def _fetch_funding_rate(symbol: str) -> dict:
    """Fetch latest funding rate from Binance Futures. Returns empty dict if unavailable."""
    try:
        resp = requests.get(
            "https://fapi.binance.com/fapi/v1/premiumIndex",
            params={"symbol": symbol},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
        fr = float(data.get("lastFundingRate", 0))
        return {
            "funding_rate": fr,
            "funding_rate_pct": fr * 100,
            "buy_adj": 1 if fr < -0.0002 else (-2 if fr > 0.0005 else 0),
            "sell_adj": 2 if fr > 0.0005 else (0 if fr > 0 else -1),
        }
    except Exception:
        return {"funding_rate": 0.0, "funding_rate_pct": 0.0, "buy_adj": 0, "sell_adj": 0}


# ------------------------------------------------------------------
# 3b) D1 BIAS + CANDLE SIGNALS
# ------------------------------------------------------------------
def compute_d1_bias(symbol: str) -> tuple[bool, bool]:
    """
    Fetch D1 klines and return (d1_bullish, d1_bearish).
    ADX-gated: only trust EMA cross when ADX > 22 (real trend, not sideways noise).
    """
    klines_d1 = _rsi_fetch_klines(symbol, "1d", limit=250)
    closes_d1 = [float(k[4]) for k in klines_d1]
    highs_d1 = [float(k[2]) for k in klines_d1]
    lows_d1 = [float(k[3]) for k in klines_d1]
    ema34_d1 = _compute_ema_series(closes_d1, 34)
    ema89_d1 = _compute_ema_series(closes_d1, 89)
    ema200_d1 = _compute_ema_series(closes_d1, 200)
    _, _, hist_d1 = _compute_macd_series(closes_d1, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    adx_data = _compute_adx(highs_d1, lows_d1, closes_d1, period=14)
    e34 = ema34_d1[-1]
    e89 = ema89_d1[-1]
    e200 = ema200_d1[-1]
    h = hist_d1[-1]
    p = closes_d1[-1]
    macd_threshold = p * 0.0001
    raw_bullish = (
        e34 is not None and e89 is not None and e34 > e89
        and (e200 is None or p > e200)
        and h is not None and h > -macd_threshold
    )
    raw_bearish = (
        e34 is not None and e89 is not None and e34 < e89
        and (e200 is None or p < e200)
        and h is not None and h < macd_threshold
    )
    # Only trust bias when market is actually trending (ADX > 22)
    is_trending = adx_data["trending"]
    d1_bullish = raw_bullish and is_trending
    d1_bearish = raw_bearish and is_trending
    return d1_bullish, d1_bearish


def compute_candle_signals(
    closes, highs, lows, rsi, macd_hist,
    ema34, ema50, ema89, ema200,
    sma_50, sma_150, bb_upper, bb_lower, stoch_k, williams_r,
    buy_zone_low, buy_zone_high, sell_zone_low, sell_zone_high,
    d1_bullish, d1_bearish, btc_rsi_h4, btc_macd_hist,
    atr_series=None, volumes=None, fng_adj=None, funding_adj=None,
) -> list[dict]:
    """
    Weighted scoring signal engine.

    BUY score (max ~14):
      +3  price inside buy zone          (base, required)
      +1  RSI < 45
      +1  RSI < 35 (extra)
      +1  MACD hist improving (hist > prev)
      +1  MACD hist > 0
      +1  EMA34 > EMA89 (trend aligned bullish)
      +1  price above SMA150 (macro support)
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

    # Pre-compute OBV signals once for the full series (used for latest candle scoring)
    _obv_signals = compute_obv_signals(closes, volumes or []) if volumes else {"buy_vol_score": 0, "sell_vol_score": 0}
    _dyn_oversold, _dyn_overbought = compute_dynamic_rsi_thresholds(rsi, lookback=100)

    for i in range(n):
        price  = closes[i]
        rsi_i  = rsi[i]       if i < len(rsi)       else None
        macd_i = macd_hist[i] if i < len(macd_hist) else None
        macd_p = macd_hist[i - 1] if i > 0 and (i - 1) < len(macd_hist) else None
        e34_i  = ema34[i]    if i < len(ema34)    else None
        e89_i  = ema89[i]    if i < len(ema89)    else None
        s150_i = sma_150[i]  if i < len(sma_150)  else None
        bb_u   = bb_upper[i]  if i < len(bb_upper)  else None
        bb_l   = bb_lower[i]  if i < len(bb_lower)  else None
        sk     = stoch_k[i]   if i < len(stoch_k)   else None
        wr_i   = williams_r[i] if i < len(williams_r) else None
        atr_i = atr_series[i] if atr_series and i < len(atr_series) else None

        # Volatility regime: skip signal in extreme conditions
        vol_extreme = False
        vol_high = False
        if atr_i is not None and price > 0:
            atr_pct = atr_i / price * 100
            if atr_pct > 5.0:
                vol_extreme = True
            elif atr_pct > 3.0:
                vol_high = True

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
            if e34_i is not None and e89_i is not None and e34_i > e89_i:
                buy_score += 1                         # EMA34 > EMA89 trend aligned
            if s150_i is not None and price > s150_i: buy_score += 1  # above SMA150
            if sk  is not None and sk  < 30:  buy_score += 1
            if wr_i is not None and wr_i < -70: buy_score += 1
            if d1_bullish:                    buy_score += 2
            if bb_l is not None and price <= bb_l: buy_score += 1
            # Dynamic RSI threshold (adaptive)
            if rsi_i is not None and rsi_i < _dyn_oversold:
                buy_score += 1
            # Volume confirmation (only apply to recent candles - last 3)
            if i >= n - 3:
                buy_score += _obv_signals["buy_vol_score"]
            # On-chain adjustments (macro, apply to all candles in zone)
            if fng_adj: buy_score += fng_adj.get("buy_adj", 0)
            if funding_adj: buy_score += funding_adj.get("buy_adj", 0)

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
            if e34_i is not None and e89_i is not None and e34_i < e89_i:
                sell_score += 1                         # EMA34 < EMA89 trend aligned
            if sk   is not None and sk   > 70:  sell_score += 1
            if wr_i is not None and wr_i > -30: sell_score += 1
            if d1_bearish:                      sell_score += 2
            if bb_u is not None and price >= bb_u: sell_score += 1
            # Dynamic RSI threshold (adaptive)
            if rsi_i is not None and rsi_i > _dyn_overbought:
                sell_score += 1
            # Volume confirmation (only apply to recent candles - last 3)
            if i >= n - 3:
                sell_score += _obv_signals["sell_vol_score"]
            # On-chain adjustments
            if fng_adj: sell_score += fng_adj.get("sell_adj", 0)
            if funding_adj: sell_score += funding_adj.get("sell_adj", 0)

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

        # Extreme volatility: suppress all signals (flash crash / pump protection)
        if vol_extreme:
            is_buy_signal = False
            is_sell_signal = False
        elif vol_high:
            # High volatility: require stronger conviction
            is_buy_signal = buy_score >= 6 and not buy_blocked
            is_sell_signal = sell_score >= 6 and not sell_blocked

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

        # ATR-based stop loss and take profit levels
        atr_levels = {}
        if atr_i and price > 0:
            mult = 1.5 if signal_strength == 3 else 2.0
            risk = mult * atr_i
            if is_buy_signal:
                atr_levels = {
                    "stop_loss": round(price - risk, 2),
                    "take_profit": round(price + 2 * risk, 2),
                    "atr": round(atr_i, 4),
                    "atr_pct": round(atr_i / price * 100, 2),
                }
            elif is_sell_signal:
                atr_levels = {
                    "stop_loss": round(price + risk, 2),
                    "take_profit": round(price - 2 * risk, 2),
                    "atr": round(atr_i, 4),
                    "atr_pct": round(atr_i / price * 100, 2),
                }

        results.append({
            "is_buy_signal":   bool(is_buy_signal),
            "is_sell_signal":  bool(is_sell_signal),
            "is_danger_zone":  bool(is_danger_zone),
            "signal_strength": signal_strength,
            "buy_score":       buy_score,
            "sell_score":      sell_score,
            "zone":            zone,
            "signal_detail":   signal_detail,
            "atr_levels":      atr_levels,
        })

    return results


# ------------------------------------------------------------------
# 3c) OPENROUTER AI ANALYSIS
# ------------------------------------------------------------------
# Cache AI analysis results for 10 minutes per (symbol, tf)
_ai_analysis_cache: Dict[str, tuple] = {}   # key -> (ts, result_str)
AI_ANALYSIS_CACHE_TTL = 600  # seconds


def call_openrouter_analysis(symbol: str, tf: str, snap: dict) -> str:
    """
    Build a market snapshot prompt and call OpenRouter API.
    Returns a Vietnamese markdown-formatted analysis string,
    or "" if the API key is not set or the call fails.
    Cached for 10 minutes per (symbol, tf) to avoid slow AI calls on every page load.
    """
    if not OPENROUTER_API_KEY:
        return ""

    cache_key = f"{symbol}_{tf}"
    now = time.time()
    entry = _ai_analysis_cache.get(cache_key)
    if entry and now - entry[0] < AI_ANALYSIS_CACHE_TTL:
        return entry[1]

    def _fmt(v, decimals=2):
        return f"{v:.{decimals}f}" if v is not None else "N/A"

    zone_label = {"buy": "Vùng MUA", "sell": "Vùng BÁN", "neutral": "Vùng trung lập"}.get(
        snap.get("zone", "neutral"), "Trung lập"
    )

    prompt = f"""Bạn là chuyên gia phân tích kỹ thuật cryptocurrency chuyên nghiệp.
Dưới đây là dữ liệu thị trường thực tế của {symbol} trên khung {tf}. Hãy phân tích và đưa ra nhận định bằng **tiếng Việt**.

---
## Snapshot thị trường — {symbol} ({tf})

| Chỉ báo | Giá trị |
|---------|---------|
| Giá hiện tại | {_fmt(snap.get('price'), 4)} USDT |
| Thay đổi ~24h | {_fmt(snap.get('change_24h'), 2)}% |
| RSI (14) | {_fmt(snap.get('rsi'))} |
| MACD Histogram | {_fmt(snap.get('macd_hist'), 6)} ({'⬆ tăng' if snap.get('macd_hist_rising') else '⬇ giảm'}) |
| EMA 34 vs EMA 89 | {'📈 EMA34 > EMA89 (tăng)' if snap.get('ema_bullish') else '📉 EMA34 < EMA89 (giảm)' if snap.get('ema_bearish') else '➡ Đan xen'} |
| Stochastic %K | {_fmt(snap.get('stoch_k'))} |
| Williams %R | {_fmt(snap.get('wr'))} |
| BB Upper / Lower | {_fmt(snap.get('bb_upper'), 4)} / {_fmt(snap.get('bb_lower'), 4)} |
| Vị trí giá | **{zone_label}** |

## Xu hướng D1 (Daily bias)
- D1 Bullish: {'✅' if snap.get('d1_bullish') else '❌'}
- D1 Bearish: {'✅' if snap.get('d1_bearish') else '❌'}

## BTC Context ({tf})
- BTC RSI: {_fmt(snap.get('btc_rsi'))}
- BTC MACD Hist: {_fmt(snap.get('btc_macd_hist'), 6)}

## Vùng giao dịch động
- 🟢 Vùng MUA: {_fmt(snap.get('buy_low'), 2)} – {_fmt(snap.get('buy_high'), 2)}
- 🔴 Vùng BÁN: {_fmt(snap.get('sell_low'), 2)} – {_fmt(snap.get('sell_high'), 2)}

## Tín hiệu bot (4H)
- Hành động: **{snap.get('tracker_action', 'N/A')}**
- Buy score: {snap.get('buy_score', 0)} / 13
- Sell score: {snap.get('sell_score', 0)} / 13

---
Viết phân tích thị trường theo đúng cấu trúc sau, bằng tiếng Việt, súc tích và chuyên nghiệp:

### 1. 📊 Xu hướng tổng quan
Mô tả xu hướng ngắn và trung hạn dựa trên EMA, D1 bias và vị trí giá trong vùng.

### 2. 🔍 Phân tích chỉ báo kỹ thuật
Nhận xét từng chỉ báo: RSI, MACD, Stochastic, Williams %R, Bollinger Bands — điểm mạnh/yếu của từng cái.

### 3. 📍 Vùng giá quan trọng
Phân tích vùng mua/bán động, mức hỗ trợ/kháng cự cần theo dõi.

### 4. 💡 Khuyến nghị giao dịch
Đưa ra khuyến nghị rõ ràng: BUY / HOLD / SELL, điều kiện vào lệnh, mức chốt lời/cắt lỗ tham khảo.

### 5. ⚠️ Rủi ro cần lưu ý
Các yếu tố có thể làm vô hiệu phân tích trên.

Không nhắc lại bảng số liệu. Dùng emoji phù hợp. Tối đa 400 từ."""

    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://qapi.app",
                "X-Title": "QAPI Crypto Dashboard",
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1500,
                "temperature": 0.3,
            },
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()["choices"][0]["message"]["content"].strip()
        _ai_analysis_cache[cache_key] = (time.time(), result)
        return result
    except Exception as e:
        logging.error(f"[OPENROUTER] Error calling {OPENROUTER_MODEL}: {e}")
        return ""


def _ai_brief_for_telegram(
    symbol: str,
    action: str,
    price: float,
    rsi: float,
    macd_hist: float,
    d1_bullish: bool,
    d1_bearish: bool,
    btc_rsi: float,
    atr_pct: float,
    interval: str,
) -> str:
    """
    Gọi AI để lấy nhận định ngắn gọn 2–3 câu cho Telegram notification.
    Trả về chuỗi plain text (không markdown), hoặc "" nếu không có API key.
    """
    if not OPENROUTER_API_KEY:
        return ""

    d1_label = "Bullish" if d1_bullish else ("Bearish" if d1_bearish else "Neutral")
    prompt = (
        f"Crypto signal alert: {symbol} — {action} tại {price:,.2f} USDT (khung {interval}).\n"
        f"RSI: {rsi:.1f} | MACD hist: {macd_hist:.4f} | D1 bias: {d1_label} | "
        f"BTC RSI: {btc_rsi:.1f} | ATR: {atr_pct:.2f}%\n\n"
        f"Viết đúng 2–3 câu bằng tiếng Việt: nhận định ngắn gọn về tín hiệu này "
        f"(xu hướng, độ tin cậy, rủi ro chính). Không dùng markdown, không bullet point, "
        f"chỉ văn xuôi thuần."
    )
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://qapi.app",
                "X-Title": "QAPI Crypto Bot",
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200,
                "temperature": 0.4,
            },
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logging.warning(f"[AI_BRIEF] Error: {e}")
        return ""


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
                _telegram_notify(f"RSI Oversold {tf} — {sym}", _fmt_dual(tf, "<30", tf_snap))
                _rsi_last_state[sym][tf] = "oversold"
            elif rsi > 70 and prev != "overbought":
                _telegram_notify(f"RSI Overbought {tf} — {sym}", _fmt_dual(tf, ">70", tf_snap))
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
    rsi_buy_threshold: float = None,
    rsi_sell_threshold: float = None,
) -> Dict[str, str]:
    """
    Quyết định BUY/SELL/HOLD với:
    - zones: (sell_low, sell_high, buy_low, buy_high, recent_low, recent_high)
    - BTC filter để tránh bán ngược trend.
    - rsi_buy/sell_threshold: dynamic thresholds (fallback to ETH_RSI_BUY/SELL env).
    """
    rsi_buy_thr = rsi_buy_threshold if rsi_buy_threshold is not None else ETH_RSI_BUY
    rsi_sell_thr = rsi_sell_threshold if rsi_sell_threshold is not None else ETH_RSI_SELL
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
        and rsi_h4 >= rsi_sell_thr
        and macd_weakening
    ):
        action = "SELL"
        reasons.append(
            f"Price {price:.1f} in SELL zone & RSI_H4 {rsi_h4:.1f} >= {rsi_sell_thr:.1f}"
        )
        reasons.append(
            f"MACD hist weakening: current {macd_hist:.4f} < prev {prev_macd_hist:.4f}"
        )

    elif buy_low <= price <= buy_high and rsi_h4 <= rsi_buy_thr:
        action = "BUY"
        reasons.append(
            f"Price {price:.1f} in BUY zone & RSI_H4 {rsi_h4:.1f} <= {rsi_buy_thr:.1f}"
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

    # Dynamic RSI thresholds — works for any coin, not just ETH
    rsi_buy_thr = rsi_sell_thr = None
    try:
        kl = _rsi_fetch_klines(symbol, interval, limit=150)
        closes_full = [float(k[4]) for k in kl]
        rsi_series = _compute_rsi_series(closes_full, RSI_PERIOD)
        rsi_buy_thr, rsi_sell_thr = compute_dynamic_rsi_thresholds(rsi_series)
    except Exception:
        pass

    decision = _eth_decide_action(
        price=price,
        rsi_h4=rsi_h4,
        macd_hist=macd_hist,
        prev_macd_hist=prev_macd_hist,
        zones=zones,
        btc_rsi_h4=btc_rsi_h4,
        btc_macd_hist=btc_macd_hist,
        btc_prev_macd_hist=btc_prev_macd_hist,
        rsi_buy_threshold=rsi_buy_thr,
        rsi_sell_threshold=rsi_sell_thr,
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

    if send_notify and action != "HOLD" and _can_notify(symbol, action):
        _mark_notified(symbol, action)
        try:
            title = f"[{action}] {symbol}"
            # Fetch ATR for stop loss levels
            try:
                kl_atr = _rsi_fetch_klines(symbol, interval, limit=50)
                h_atr = [float(k[2]) for k in kl_atr]
                l_atr = [float(k[3]) for k in kl_atr]
                c_atr = [float(k[4]) for k in kl_atr]
                atr_s = _compute_atr_series(h_atr, l_atr, c_atr)
                atr_val = next((v for v in reversed(atr_s) if v is not None), None)
            except Exception:
                atr_val = None

            msg_lines = [
                f"💰 Giá: <b>{price:,.2f}</b> USDT",
                f"📊 RSI H4: {rsi_h4:.1f} | MACD Hist: {macd_hist:.4f}",
                f"🎯 Zone: BUY[{buy_low:.1f}-{buy_high:.1f}] SELL[{sell_low:.1f}-{sell_high:.1f}]",
                f"₿ BTC RSI: {btc_rsi_h4:.1f} | BTC MACD: {btc_macd_hist:.4f}",
            ]
            atr_pct_val = 0.0
            if atr_val:
                mult = 2.0
                risk = mult * atr_val
                atr_pct_val = atr_val / price * 100
                if action == "BUY":
                    sl = price - risk
                    tp = price + 2 * risk
                    msg_lines.append(f"🛑 SL: {sl:,.2f} | 🎯 TP: {tp:,.2f} (ATR×{mult})")
                else:
                    sl = price + risk
                    tp = price - 2 * risk
                    msg_lines.append(f"🛑 SL: {sl:,.2f} | 🎯 TP: {tp:,.2f} (ATR×{mult})")
                msg_lines.append(f"📐 ATR: {atr_val:.2f} ({atr_pct_val:.2f}%)")
            msg_lines.append(f"⏰ {now_utc}")

            # AI brief (2-3 câu nhận định ngắn)
            d1_bull, d1_bear = False, False
            try:
                d1_bull, d1_bear = compute_d1_bias(symbol)
            except Exception:
                pass
            ai_brief = _ai_brief_for_telegram(
                symbol=symbol,
                action=action,
                price=price,
                rsi=rsi_h4,
                macd_hist=macd_hist,
                d1_bullish=d1_bull,
                d1_bearish=d1_bear,
                btc_rsi=btc_rsi_h4,
                atr_pct=atr_pct_val,
                interval=interval,
            )
            if ai_brief:
                msg_lines.append(f"\n🤖 <i>{ai_brief}</i>")

            _telegram_notify(title, "\n".join(msg_lines))
        except Exception as e:
            logging.error(f"[SYMBOL_TRACKER_NOTIFY] Error: {e}")

    return payload


def save_signal_to_db(
    symbol: str,
    timeframe: str,
    signal_type: str,
    price: float,
    rsi: Optional[float] = None,
    macd_hist: Optional[float] = None,
    buy_score: int = 0,
    sell_score: int = 0,
    signal_strength: int = 1,
    signal_detail: str = "",
) -> None:
    """
    Persist a BUY or SELL signal to the signal_history table.
    Duplicate-guard: skip if the same symbol+timeframe+signal_type was saved
    within NOTIFY_COOLDOWN_MINUTES (prevents duplicate DB rows from reruns).
    """
    try:
        from datetime import timezone, timedelta
        cutoff_dt = datetime.now(tz=timezone.utc) - timedelta(minutes=NOTIFY_COOLDOWN_MINUTES)
        cutoff = cutoff_dt.isoformat()

        # Duplicate-guard: skip if same signal fired within last 30 minutes
        check = (
            supabase_admin.table("signal_history")
            .select("id")
            .eq("symbol", symbol)
            .eq("timeframe", timeframe)
            .eq("signal_type", signal_type)
            .gte("created_at", cutoff)
            .limit(1)
            .execute()
        )
        if check.data:
            logging.info(f"[SIGNAL_DB] Skip duplicate {signal_type} {symbol} (within {NOTIFY_COOLDOWN_MINUTES}m)")
            return

        supabase_admin.table("signal_history").insert({
            "symbol":          symbol,
            "timeframe":       timeframe,
            "signal_type":     signal_type,
            "price":           price,
            "rsi":             rsi,
            "macd_hist":       macd_hist,
            "buy_score":       buy_score,
            "sell_score":      sell_score,
            "signal_strength": signal_strength,
            "signal_detail":   signal_detail,
        }).execute()
        logging.info(f"[SIGNAL_DB] Saved {signal_type} {symbol} @ {price}")
    except Exception as e:
        logging.error(f"[SIGNAL_DB] Error saving signal: {e}")


def symbols_tracker_job():
    """
    Job chạy mỗi 10 phút:
    - Lấy danh sách symbol is_active = true trong bot_subscriptions
    - Mỗi symbol → run_symbol_tracker_once(send_notify=True)
    - BUY/SELL signals are persisted to signal_history table
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

    # Fetch market sentiment once per job run
    try:
        fng = _fetch_fear_greed()
        logging.info(f"[TRACKER_JOB] F&G Index: {fng['value']}")
    except Exception:
        pass

    for row in rows:
        symbol = (row.get("symbol") or "").upper()
        if not symbol:
            continue
        try:
            payload = run_symbol_tracker_once(symbol, send_notify=True)
            action = payload["action"]
            logging.info(
                f"[SYMBOL_TRACKER_JOB] {symbol}: action={action} price={payload['price']}"
            )
            if action in ("BUY", "SELL"):
                save_signal_to_db(
                    symbol=symbol,
                    timeframe=payload.get("timeframe", TRACKER_INTERVAL),
                    signal_type=action,
                    price=payload["price"],
                    rsi=payload.get("rsi_h4"),
                    macd_hist=payload.get("macd_hist"),
                    signal_detail=payload.get("reason", ""),
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
    valid_tfs = {"15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}
    if tf not in valid_tfs:
        tf = "4h"
    interval = valid_tfs[tf]

    # ── Parallel network fetches — all independent calls run concurrently ────
    def _safe_klines():
        try:
            return _rsi_fetch_klines(symbol, interval, limit=200)
        except Exception as e:
            logging.error(f"[SYMBOL DASH] Error fetching klines for {symbol}: {e}")
            return []

    def _safe_d1_bias():
        try:
            return compute_d1_bias(symbol)
        except Exception as e:
            logging.error(f"[SYMBOL DASH] Error computing D1 bias for {symbol}: {e}")
            return (False, False)

    def _safe_btc_context():
        try:
            _, btc_rsi = _rsi_latest("BTCUSDT", interval, RSI_PERIOD)
            _, _, btc_macd_h, _ = _macd_latest_with_prev("BTCUSDT", interval)
            return btc_rsi, btc_macd_h
        except Exception as e:
            logging.error(f"[SYMBOL DASH] Error fetching BTC context: {e}")
            return None, None

    klines, d1_result, btc_result, fng_data, funding_data = await asyncio.gather(
        asyncio.to_thread(_safe_klines),
        asyncio.to_thread(_safe_d1_bias),
        asyncio.to_thread(_safe_btc_context),
        asyncio.to_thread(_fetch_fear_greed),
        asyncio.to_thread(_fetch_funding_rate, symbol),
    )
    d1_bullish, d1_bearish = d1_result
    btc_rsi_h4, btc_macd_hist_val = btc_result
    # ─────────────────────────────────────────────────────────────────

    labels: List[str] = []
    closes: List[float] = []
    highs: List[float] = []
    lows: List[float] = []
    volumes: List[float] = []

    for k in klines:
        try:
            open_time_ms = int(k[0])
            dt = datetime.utcfromtimestamp(open_time_ms / 1000.0)
            labels.append(dt.strftime("%Y-%m-%d %H:%M"))

            highs.append(float(k[2]))
            lows.append(float(k[3]))
            closes.append(float(k[4]))
            volumes.append(float(k[5]))
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
    ema34 = _compute_ema_series(closes, 34)
    ema50 = _compute_ema_series(closes, 50)
    ema89 = _compute_ema_series(closes, 89)
    ema200 = _compute_ema_series(closes, 200)
    sma_50 = _sma_series(closes, 50)
    sma_150 = _sma_series(closes, 150)

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
        len(volumes),
        len(rsi_values),
        len(macd_hist_values),
        len(ema34),
        len(ema50),
        len(ema89),
        len(ema200),
        len(sma_50),
        len(sma_150),
        len(bb_upper),
        len(bb_lower),
        len(stoch_k),
        len(williams_r_vals),
    )

    labels = labels[-min_len:]
    closes_trimmed = closes[-min_len:]
    highs_trimmed = highs[-min_len:]
    lows_trimmed = lows[-min_len:]
    volumes_trimmed = volumes[-min_len:]
    rsi_values = rsi_values[-min_len:]
    macd_hist_values = macd_hist_values[-min_len:]
    ema34 = ema34[-min_len:]
    ema50 = ema50[-min_len:]
    ema89 = ema89[-min_len:]
    ema200 = ema200[-min_len:]
    sma_50 = sma_50[-min_len:]
    sma_150 = sma_150[-min_len:]
    bb_middle = bb_middle[-min_len:]
    bb_upper = bb_upper[-min_len:]
    bb_lower = bb_lower[-min_len:]
    stoch_k = stoch_k[-min_len:]
    williams_r_vals = williams_r_vals[-min_len:]

    # ATR series for volatility regime + stop loss
    atr_vals = _compute_atr_series(highs_trimmed, lows_trimmed, closes_trimmed)
    # OBV signals
    obv_sig = compute_obv_signals(closes_trimmed, volumes_trimmed)
    # fng_data and funding_data already fetched in parallel above

    rows_json: List[Dict[str, Any]] = []
    for i in range(min_len):
        rows_json.append(
            {
                "time_str": labels[i],
                "price": closes_trimmed[i],
                "high": highs_trimmed[i],
                "low": lows_trimmed[i],
                "volume": volumes_trimmed[i],
                "rsi_h4": rsi_values[i],
                "macd_hist": macd_hist_values[i],
                "ema34": ema34[i],
                "ema50": ema50[i],
                "ema89": ema89[i],
                "ema200": ema200[i],
                "sma_50": sma_50[i],
                "sma_150": sma_150[i],
                "bb_middle": bb_middle[i],
                "bb_upper": bb_upper[i],
                "bb_lower": bb_lower[i],
                "stoch_k": stoch_k[i],
                "wr": williams_r_vals[i],
                "atr": atr_vals[i] if i < len(atr_vals) else None,
                "obv_trend": obv_sig["obv_trend"],
            }
        )

    # d1_bullish, d1_bearish, btc_rsi_h4, btc_macd_hist_val already set from parallel gather above

    # --- Compute per-candle signals ---
    signals = compute_candle_signals(
        closes=closes_trimmed,
        highs=highs_trimmed,
        lows=lows_trimmed,
        rsi=rsi_values,
        macd_hist=macd_hist_values,
        ema34=ema34,
        ema50=ema50,
        ema89=ema89,
        ema200=ema200,
        sma_50=sma_50,
        sma_150=sma_150,
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
        atr_series=atr_vals,
        volumes=volumes_trimmed,
        fng_adj=fng_data,
        funding_adj=funding_data,
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

    # --- Historical DB signals ---
    db_signals: List[Dict[str, Any]] = []
    try:
        sig_resp = (
            supabase_admin.table("signal_history")
            .select("signal_type, price, rsi, macd_hist, buy_score, sell_score, signal_strength, signal_detail, created_at")
            .eq("symbol", symbol)
            .order("created_at", desc=False)
            .limit(500)
            .execute()
        )
        db_signals = sig_resp.data or []
    except Exception as e:
        logging.error(f"[SYMBOL DASH] Error fetching signal_history: {e}")

    db_signals_json = json.dumps(db_signals)

    # --- OpenRouter AI market analysis ---
    last_signal = signals[-1] if signals else {}
    ai_snap = {
        "price":           last_price,
        "change_24h":      change_24h,
        "rsi":             last_rsi,
        "macd_hist":       macd_hist_values[-1] if macd_hist_values else None,
        "macd_hist_rising": (
            len(macd_hist_values) >= 2
            and macd_hist_values[-1] is not None
            and macd_hist_values[-2] is not None
            and macd_hist_values[-1] > macd_hist_values[-2]
        ),
        "ema_bullish":     (ema34[-1] is not None and ema89[-1] is not None and ema34[-1] > ema89[-1]),
        "ema_bearish":     (ema34[-1] is not None and ema89[-1] is not None and ema34[-1] < ema89[-1]),
        "stoch_k":         stoch_k[-1] if stoch_k else None,
        "wr":              williams_r_vals[-1] if williams_r_vals else None,
        "bb_upper":        bb_upper[-1] if bb_upper else None,
        "bb_lower":        bb_lower[-1] if bb_lower else None,
        "d1_bullish":      d1_bullish,
        "d1_bearish":      d1_bearish,
        "btc_rsi":         btc_rsi_h4,
        "btc_macd_hist":   btc_macd_hist_val,
        "buy_low":         buy_low,
        "buy_high":        buy_high,
        "sell_low":        sell_low,
        "sell_high":       sell_high,
        "zone":            last_signal.get("zone", "neutral"),
        "tracker_action":  tracker_action,
        "buy_score":       last_signal.get("buy_score", 0),
        "sell_score":      last_signal.get("sell_score", 0),
    }
    ai_analysis = call_openrouter_analysis(symbol, tf, ai_snap)

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
        "ai_analysis": ai_analysis,
        "db_signals_json": db_signals_json,
        "fng_value": fng_data.get("value", 50),
        "funding_rate_pct": funding_data.get("funding_rate_pct", 0.0),
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
    )
