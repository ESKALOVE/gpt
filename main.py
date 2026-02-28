"""Basic Gate.io futures bot loop using ccxt and an RSI entry strategy."""

import time

import config
from exchange import GateioFuturesClient
from strategy import calculate_rsi, generate_signal


def run_bot():
    client = GateioFuturesClient(config.API_KEY, config.API_SECRET)
    in_position = False
    entry_price = None

    while True:
        try:
            candles = client.fetch_ohlcv(
                config.SYMBOL, timeframe=config.TIMEFRAME, limit=config.LIMIT
            )
            closes = [candle[4] for candle in candles]
            last_price = closes[-1]

            if not in_position:
                rsi = calculate_rsi(closes, period=config.RSI_PERIOD)
                signal = generate_signal(
                    rsi,
                    oversold=config.RSI_OVERSOLD,
                    overbought=config.RSI_OVERBOUGHT,
                )

                print(f"Price: {last_price}")
                print(f"RSI: {rsi:.2f}" if rsi is not None else "RSI: N/A")
                print(f"Signal: {signal}")

                if signal == "buy":
                    if config.LIVE_TRADING:
                        order = client.create_market_order(
                            config.SYMBOL, "buy", config.ORDER_SIZE
                        )
                        print("Entry order placed:", order)
                    else:
                        print("[SIM] Would place market BUY order")

                    in_position = True
                    entry_price = last_price
                    print(f"Entered long at {entry_price}")
            else:
                take_profit_price = entry_price * (1 + config.TAKE_PROFIT_PCT)
                stop_loss_price = entry_price * (1 - config.STOP_LOSS_PCT)

                print(f"Price: {last_price}")
                print(f"In position from: {entry_price}")
                print(f"TP: {take_profit_price} | SL: {stop_loss_price}")

                exit_reason = None
                if last_price >= take_profit_price:
                    exit_reason = "take profit"
                elif last_price <= stop_loss_price:
                    exit_reason = "stop loss"

                if exit_reason:
                    if config.LIVE_TRADING:
                        order = client.create_market_order(
                            config.SYMBOL, "sell", config.ORDER_SIZE
                        )
                        print(f"Exit order placed ({exit_reason}):", order)
                    else:
                        print(f"[SIM] Would place market SELL order ({exit_reason})")

                    in_position = False
                    entry_price = None

            time.sleep(config.POLL_SECONDS)

        except Exception as exc:
            print("Bot error:", exc)
            time.sleep(config.POLL_SECONDS)


if __name__ == "__main__":
    run_bot()
