"""Exchange wrapper using ccxt for Gate.io futures."""

import ccxt


class GateioFuturesClient:
    def __init__(self, api_key: str, api_secret: str):
        self.exchange = ccxt.gateio(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": "swap"},
            }
        )

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "5m",
        limit: int | None = 100,
        since: int | None = None,
        params: dict | None = None,
    ):
        return self.exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            limit=limit,
            since=since,
            params=params or {},
        )

    def create_market_order(self, symbol: str, side: str, amount: float):
        return self.exchange.create_order(symbol, "market", side, amount)
