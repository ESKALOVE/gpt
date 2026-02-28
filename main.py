"""Basic Gate.io futures trading bot loop using ccxt and an RSI strategy."""

import time

import config
from exchange import GateioFuturesClient
from strategy import calculate_rsi, generate_signal


def run_bot():
    client = GateioFuturesClient(config.API_KEY, config.API_SECRET)

    while True:
        try:
            candles = client.fetch_ohlcv(
                config.SYMBOL, timeframe=config.TIMEFRAME, limit=config.LIMIT
            )
            closes = [candle[4] for candle in candles]

            rsi = calculate_rsi(closes, period=config.RSI_PERIOD)
            signal = generate_signal(
                rsi,
                oversold=config.RSI_OVERSOLD,
                overbought=config.RSI_OVERBOUGHT,
            )

            print(f"RSI: {rsi:.2f}" if rsi is not None else "RSI: N/A")
            print(f"Signal: {signal}")

            if config.LIVE_TRADING and signal in {"buy", "sell"}:
                order = client.create_market_order(
                    config.SYMBOL, signal, config.ORDER_SIZE
                )
                print("Order placed:", order)

            time.sleep(config.POLL_SECONDS)

        except Exception as exc:
            print("Bot error:", exc)
            time.sleep(config.POLL_SECONDS)


if __name__ == "__main__":
    run_bot()
