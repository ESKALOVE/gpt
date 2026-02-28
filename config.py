"""Configuration for the Gate.io futures trading bot."""

API_KEY = "YOUR_GATEIO_API_KEY"
API_SECRET = "YOUR_GATEIO_API_SECRET"

SYMBOL = "BTC/USDT:USDT"
TIMEFRAME = "5m"
LIMIT = 100

RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

ORDER_SIZE = 0.001
POLL_SECONDS = 30

# Set True to send real orders. Keep False while testing.
LIVE_TRADING = False
