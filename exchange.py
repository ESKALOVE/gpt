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
        kwargs = {"timeframe": timeframe, "params": params or {}}
        if since is not None:
            kwargs["since"] = since
        # Gate.io 선물처럼 from/to를 쓰는 경우 limit를 함께 보내면 오류가 날 수 있어
        # limit가 None일 때는 아예 인자를 전달하지 않습니다.
        if limit is not None:
            kwargs["limit"] = limit
        return self.exchange.fetch_ohlcv(symbol, **kwargs)

    def create_market_order(self, symbol: str, side: str, amount: float):
        return self.exchange.create_order(symbol, "market", side, amount)
