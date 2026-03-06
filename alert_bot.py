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

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
EXCHANGE_NAME = os.getenv("EXCHANGE", "binance").strip()
SYMBOL = os.getenv("SYMBOL", "ETH/USDT").strip()
CHECK_INTERVAL_SEC = int(os.getenv("CHECK_INTERVAL_SEC", "60"))

# Trend D1 + Entry H4
TF_REGIME = "1d"
TF_ENTRY = "4h"

ANTI_MISS_RSI_D1 = float(os.getenv("ANTI_MISS_RSI_D1", "55"))
ANTI_MISS_DIST_BARS = int(os.getenv("ANTI_MISS_DIST_BARS", "6"))

SWING_HIGH_LOOKBACK_H4 = int(os.getenv("SWING_HIGH_LOOKBACK_H4", "36"))
SWING_HIGH_ATR_MULT = float(os.getenv("SWING_HIGH_ATR_MULT", "0.45"))

TG_POLL_TIMEOUT_SEC = int(os.getenv("TG_POLL_TIMEOUT_SEC", "0"))

STATE_FILE = "bot_state.json"


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
            print("[TELEGRAM ERROR] Missing TELEGRAM_CHAT_ID")
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
    payload = {"timeout": TG_POLL_TIMEOUT_SEC}
    if offset is not None:
        payload["offset"] = offset
    try:
        return tg_api("getUpdates", payload, timeout=25)
    except Exception as e:
        print(f"[TELEGRAM ERROR] getUpdates: {e}")
        return {"ok": False, "result": []}


def load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {
            "last_signal_entry_ts": None,
            "stop_sell_mode": False,
            "last_mode_regime_ts": None,
            "tg_update_offset": None,
        }
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {
            "last_signal_entry_ts": None,
            "stop_sell_mode": False,
            "last_mode_regime_ts": None,
            "tg_update_offset": None,
        }


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def slope(series: pd.Series, lookback: int = 12) -> float:
    s = series.dropna().tail(lookback)
    if len(s) < lookback:
        return 0.0
    y = s.values
    x = np.arange(len(y))
    return float(np.polyfit(x, y, 1)[0])


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


@dataclass
class Regime:
    ok: bool
    score: int
    details: Dict[str, bool]
    ema200_slope: float


def compute_regime(df_d1: pd.DataFrame, df_btc_d1: pd.DataFrame) -> Regime:
    c = df_d1["close"]
    ema50 = ema(c, 50)
    ema200 = ema(c, 200)

    btc_c = df_btc_d1["close"]
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
        "BTC_confirm": cond4,
    }
    score = sum(details.values())
    ok = score >= 4
    return Regime(ok=ok, score=score, details=details, ema200_slope=slope(ema200, 12))


def compute_stop_sell_mode(df_d1: pd.DataFrame) -> Tuple[bool, str]:
    c = df_d1["close"]
    ema50 = ema(c, 50)
    ema200 = ema(c, 200)
    rsi_d1 = rsi(c, 14)

    close_now = float(c.iloc[-1])
    ema200_now = float(ema200.iloc[-1])

    if close_now > ema200_now:
        return True, f"Close(D1) {close_now:.2f} > EMA200(D1) {ema200_now:.2f}"

    dist = (ema200 - ema50).dropna().tail(ANTI_MISS_DIST_BARS + 1)
    shrinking = False
    if len(dist) >= ANTI_MISS_DIST_BARS + 1:
        shrinking = float(dist.iloc[-1]) < float(dist.iloc[0])

    if shrinking and float(rsi_d1.iloc[-1]) > ANTI_MISS_RSI_D1:
        return True, f"Dist(EMA200-EMA50) shrinking + RSI(D1) {float(rsi_d1.iloc[-1]):.1f} > {ANTI_MISS_RSI_D1}"

    return False, ""


def fib_zone_from_swing(df: pd.DataFrame, lookback: int = 48) -> Tuple[float, float]:
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


def swing_high_zone(df_h4: pd.DataFrame, lookback: int, atr_h4: float, atr_mult: float) -> Tuple[float, float, float]:
    sub = df_h4.tail(lookback)
    sh = float(sub["high"].max())
    width = max(atr_h4 * atr_mult, sh * 0.0015)
    return sh, sh - width, sh + width


def compute_zones(df_d1: pd.DataFrame, df_h4: pd.DataFrame) -> Dict[str, Any]:
    price = float(df_h4["close"].iloc[-1])

    ema50_h4 = float(ema(df_h4["close"], 50).iloc[-1])
    ema200_h4 = float(ema(df_h4["close"], 200).iloc[-1])
    ema50_d1 = float(ema(df_d1["close"], 50).iloc[-1])
    ema200_d1 = float(ema(df_d1["close"], 200).iloc[-1])

    atr_series = atr(df_h4, 14)
    atr_h4 = float(atr_series.iloc[-1]) if not np.isnan(atr_series.iloc[-1]) else 0.0

    fib_lo, fib_hi = fib_zone_from_swing(df_h4, lookback=48)
    support = float(df_h4["low"].tail(36).min())

    w_sell_h4 = max(atr_h4 * 0.35, price * 0.0015)
    w_sell_d1 = max(atr_h4 * 0.60, price * 0.0020)
    w_buy = max(atr_h4 * 0.80, price * 0.0020)

    sell_zone_candidates = [
        ("EMA50_H4", ema50_h4 - w_sell_h4, ema50_h4 + w_sell_h4),
        ("EMA50_D1", ema50_d1 - w_sell_d1, ema50_d1 + w_sell_d1),
    ]
    if not np.isnan(fib_lo) and not np.isnan(fib_hi):
        sell_zone_candidates.append(("FIB_0.5_0.618", fib_lo, fib_hi))

    passed_all = True
    for _, lo, hi in sell_zone_candidates:
        if price <= hi:
            passed_all = False
            break

    swing_info = None
    if passed_all and atr_h4 > 0:
        sh, zlo, zhi = swing_high_zone(
            df_h4,
            lookback=SWING_HIGH_LOOKBACK_H4,
            atr_h4=atr_h4,
            atr_mult=SWING_HIGH_ATR_MULT,
        )
        sell_zone_candidates.append(("SWING_HIGH_SUPPLY", zlo, zhi))
        swing_info = {
            "swing_high": sh,
            "lookback": SWING_HIGH_LOOKBACK_H4,
            "atr_mult": SWING_HIGH_ATR_MULT,
        }

    buy_zone = (support - w_buy, support + w_buy)

    return {
        "price": price,
        "ema50_h4": ema50_h4,
        "ema200_h4": ema200_h4,
        "ema50_d1": ema50_d1,
        "ema200_d1": ema200_d1,
        "atr_h4": atr_h4,
        "fib_zone": (fib_lo, fib_hi),
        "support": support,
        "sell_zones": sell_zone_candidates,
        "buy_zone": buy_zone,
        "passed_all_base_zones": passed_all,
        "swing_info": swing_info,
    }


def check_sell(df_d1: pd.DataFrame, df_h4: pd.DataFrame, regime: Regime, stop_sell_mode: bool) -> Tuple[bool, str]:
    if not regime.ok or stop_sell_mode:
        return False, "Regime not OK or STOP_SELL_MODE"

    zones = compute_zones(df_d1, df_h4)
    price = zones["price"]
    rsi_h4 = float(rsi(df_h4["close"], 14).iloc[-1])

    low36 = float(df_h4["low"].tail(36).min())
    retrace36 = (price - low36) / low36 * 100 if low36 > 0 else 0.0

    in_zone = False
    zone_name = None
    for name, lo, hi in zones["sell_zones"]:
        if lo <= price <= hi:
            in_zone = True
            zone_name = name
            break

    if retrace36 >= 4.0 and rsi_h4 > 60.0 and in_zone:
        return True, f"Retrace36={retrace36:.1f}% RSI(H4)={rsi_h4:.1f} InZone={zone_name}"

    return False, "No SELL setup"


def check_buy(df_h4: pd.DataFrame, df_d1: pd.DataFrame) -> Tuple[bool, str]:
    c = df_h4["close"]
    price = float(c.iloc[-1])
    rsi_h4 = float(rsi(c, 14).iloc[-1])

    high12 = float(df_h4["high"].tail(12).max())
    drop12 = (high12 - price) / high12 * 100 if high12 > 0 else 0.0

    trigger = (rsi_h4 < 40.0) or (drop12 >= 6.0)

    ema200_h4 = float(ema(c, 200).iloc[-1])
    atr_h4 = float(atr(df_h4, 14).iloc[-1])
    zones = compute_zones(df_d1, df_h4)
    bz_lo, bz_hi = zones["buy_zone"]
    in_buy_zone = (bz_lo <= price <= bz_hi)

    near_ema200 = price <= (ema200_h4 + 0.25 * atr_h4)

    if trigger and (near_ema200 or in_buy_zone):
        return True, f"BUY: trigger=True (RSI={rsi_h4:.1f}, Drop12={drop12:.2f}%), nearEMA200={near_ema200}, inBuyZone={in_buy_zone}"

    return False, f"No BUY: trigger={trigger} (RSI={rsi_h4:.1f}, Drop12={drop12:.2f}%), nearEMA200={near_ema200}, inBuyZone={in_buy_zone}"


def format_check_report(df_d1: pd.DataFrame, df_h4: pd.DataFrame, df_btc_d1: pd.DataFrame) -> str:

    regime = compute_regime(df_d1, df_btc_d1)
    stop_mode, _ = compute_stop_sell_mode(df_d1)
    zones = compute_zones(df_d1, df_h4)

    price = zones["price"]

    ema50_h4 = zones["ema50_h4"]
    ema200_h4 = zones["ema200_h4"]

    bz_lo, bz_hi = zones["buy_zone"]

    # tìm sell zone chính
    sell_lo, sell_hi = None, None
    for name, lo, hi in zones["sell_zones"]:
        if name == "EMA50_H4":
            sell_lo, sell_hi = lo, hi
            break

    if sell_lo is None:
        name, sell_lo, sell_hi = zones["sell_zones"][0]

    # vị trí hiện tại
    if price > ema50_h4:
        position = "trên EMA50 H4"
    elif price > ema200_h4:
        position = "giữa EMA50–EMA200"
    else:
        position = "dưới EMA200"

    # trend text
    if regime.score == 4:
        trend = "Giảm mạnh"
    elif regime.score == 3:
        trend = "Giảm nhẹ"
    else:
        trend = "Không rõ"

    # tín hiệu
    sell_ok, _ = check_sell(df_d1, df_h4, regime, stop_mode)
    buy_ok, _ = check_buy(df_h4, df_d1)

    action = "ĐỨNG CHỜ"

    if sell_ok:
        action = "🔴 CÂN NHẮC BÁN"

    elif buy_ok:
        action = "🟢 CÂN NHẮC MUA"

    elif stop_mode:
        action = "⛔ TẠM KHÔNG BÁN"

    report = ""
    report += f"📊 {SYMBOL}: {price:.0f}\n\n"
    report += f"Trend D1: {trend} ({regime.score}/4)\n"
    report += f"Vị trí: {position}\n"
    report += f"Sell zone: {sell_lo:.0f}–{sell_hi:.0f}\n"
    report += f"Buy zone: {bz_lo:.0f}–{bz_hi:.0f}\n\n"
    report += f"👉 Hành động: {action}"

    return report


def handle_telegram_commands(ex, state: dict) -> None:
    offset = state.get("tg_update_offset", None)
    resp = tg_get_updates(offset)
    if not resp.get("ok", False):
        return

    updates: List[dict] = resp.get("result", [])
    if not updates:
        return

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
                df_d1 = fetch_df(ex, SYMBOL, TF_REGIME, limit=300)
                df_h4 = fetch_df(ex, SYMBOL, TF_ENTRY, limit=400)
                df_btc_d1 = fetch_df(ex, "BTC/USDT", TF_REGIME, limit=300)

                report = format_check_report(df_d1, df_h4, df_btc_d1)
                tg_send(report, chat_id=chat_id)
            except Exception as e:
                tg_send(f"⚠️ /check error: {type(e).__name__}: {e}", chat_id=chat_id)


def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN")
    if not SYMBOL:
        raise RuntimeError("Missing SYMBOL")

    ex = make_exchange()
    state = load_state()

    if TELEGRAM_CHAT_ID:
        tg_send(f"Bot started for {SYMBOL}\nMode: Trend D1 + Entry H4\nCommands: /check")

    last_signal_entry_ts = state.get("last_signal_entry_ts")
    stop_mode_prev = bool(state.get("stop_sell_mode", False))
    last_mode_regime_ts = state.get("last_mode_regime_ts")

    while True:
        try:
            handle_telegram_commands(ex, state)

            df_d1 = fetch_df(ex, SYMBOL, TF_REGIME, limit=300)
            df_h4 = fetch_df(ex, SYMBOL, TF_ENTRY, limit=400)
            df_btc_d1 = fetch_df(ex, "BTC/USDT", TF_REGIME, limit=300)

            regime = compute_regime(df_d1, df_btc_d1)
            stop_mode, stop_reason = compute_stop_sell_mode(df_d1)

            regime_ts = df_d1.index[-1].isoformat()
            if stop_mode != stop_mode_prev and last_mode_regime_ts != regime_ts and TELEGRAM_CHAT_ID:
                if stop_mode:
                    tg_send(f"🟠 STOP SELL MODE\nReason: {stop_reason}")
                else:
                    tg_send("🟢 SELL MODE RESUMED")
                stop_mode_prev = stop_mode
                last_mode_regime_ts = regime_ts
                state["stop_sell_mode"] = stop_mode
                state["last_mode_regime_ts"] = regime_ts
                save_state(state)

            sell_ok, sell_reason = check_sell(df_d1, df_h4, regime, stop_mode)
            buy_ok, buy_reason = check_buy(df_h4, df_d1)

            entry_ts = df_h4.index[-1].isoformat()

            if entry_ts != last_signal_entry_ts and TELEGRAM_CHAT_ID:
                price = float(df_h4["close"].iloc[-1])

                if sell_ok:
                    tg_send(
                        f"🚨 SELL SIGNAL {SYMBOL}\n"
                        f"Price: {price:.2f}\n"
                        f"Mode: Trend D1 + Entry H4\n"
                        f"Regime score: {regime.score}/4 | STOP_SELL_MODE={stop_mode}\n"
                        f"Reason: {sell_reason}"
                    )
                    last_signal_entry_ts = entry_ts

                elif buy_ok:
                    tg_send(
                        f"✅ BUY SIGNAL {SYMBOL}\n"
                        f"Price: {price:.2f}\n"
                        f"Mode: Trend D1 + Entry H4\n"
                        f"Reason: {buy_reason}"
                    )
                    last_signal_entry_ts = entry_ts

                state["last_signal_entry_ts"] = last_signal_entry_ts
                save_state(state)

            time.sleep(CHECK_INTERVAL_SEC)

        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}")
            time.sleep(15)


if __name__ == "__main__":
    main()
