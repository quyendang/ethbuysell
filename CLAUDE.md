# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Development
uvicorn main:app --reload

# Production
uvicorn main:app --host 0.0.0.0 --port 8000

# Docker
docker build -t ethbuysell .
docker run -e SUPABASE_URL=... -e SUPABASE_KEY=... -e SUPABASE_SERVICE_ROLE_KEY=... ethbuysell
```

## Environment Variables

**Required:**
```bash
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=eyJ...
SUPABASE_SERVICE_ROLE_KEY=eyJ...
```

**Optional (with defaults):**
```bash
RSI_SYMBOLS=ETHUSDT,BTCUSDT
ETH_SELL_ZONE_LOW=3650
ETH_SELL_ZONE_HIGH=3700
ETH_BUY_ZONE_LOW=3350
ETH_BUY_ZONE_HIGH=3450
ETH_RSI_SELL=65
ETH_RSI_BUY=40
OPENROUTER_API_KEY=...
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_IDS=123456789,987654321   # comma-separated list of chat IDs
```

## Architecture

Single-file FastAPI app (`main.py`, ~1,475 lines) with one HTML template (`templates/symbol_dashboard.html`).

### Data Flow

1. **Background scheduler** (`APScheduler`) runs `symbols_tracker_job()` every 10 minutes: fetches active symbols from `bot_subscriptions` table → computes signals → saves to `signal_history` → sends Pushover notifications.

2. **Dashboard endpoint** `GET /bots/{symbol}?tf={timeframe}` (15m/1h/4h/1d): fetches OHLCV from Binance API → computes all indicators → generates signal → queries `signal_history` from Supabase → calls OpenRouter AI → renders Jinja2 template.

3. **Subscribe/unsubscribe** `POST /bots/subscribe` and `POST /bots/unsubscribe`: toggle `bot_subscriptions.is_active` for a symbol.

### Technical Indicators (all computed in-memory from Binance klines)

- RSI (Wilder's method, period 14)
- MACD (fast/slow/signal: 12/26/9)
- EMA 34, 50, 89, 200
- SMA 50, 150
- Bollinger Bands (period 20, k=2.0)
- Stochastic Oscillator %K/%D
- Williams %R
- D1 daily bias (bullish/bearish/neutral)

### Signal Generation (`compute_candle_signals()`)

Multi-gate scoring system:
1. Price must be in buy/sell zone
2. RSI crosses threshold
3. MACD histogram direction
4. EMA 34 vs EMA 89 alignment
5. D1 daily bias filter
6. BTC RSI correlation

Returns `signal_strength` 1–3 based on how many gates pass.

### Database (Supabase/PostgreSQL)

Two tables:
- **`bot_subscriptions`** — `symbol` (PK), `is_active`, `updated_at`
- **`signal_history`** — `symbol`, `timeframe`, `signal_type` (BUY/SELL), `price`, `rsi`, `macd_hist`, `buy_score`, `sell_score`, `signal_strength`, `signal_detail`, `created_at`

The app uses two Supabase clients: standard key for reads, service role key for writes/deletes. Signals have a 30-minute duplicate guard in `save_signal_to_db()`.

### Frontend

`symbol_dashboard.html` uses Chart.js to render 5 interactive charts: candlestick+EMAs+BB, RSI, MACD histogram, volume, and stochastic. Buy/sell zone shading and BUY/SELL signal triangles are rendered as custom Chart.js plugins. AI analysis is markdown-rendered via Marked.js.
