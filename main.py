"""Gate.io futures bot with RSI + Bollinger re-entry + ADX filter."""

import time

import config
from exchange import GateioFuturesClient
from strategy import generate_signal


def _count_candles_since(candles, since_ts):
    """since_ts 이후 확정된 캔들 개수를 세어 타임스탑/쿨다운 계산에 사용합니다."""
    return sum(1 for c in candles if c[0] > since_ts)


def run_bot():
    client = GateioFuturesClient(config.API_KEY, config.API_SECRET)

    # 포지션 상태: 양방향(롱/숏) 지원
    in_position = False
    side = None
    entry_price = None
    entry_index = None  # 진입 시점 캔들 timestamp

    # 손절 발생 후 일정 캔들 동안 신규 진입을 막기 위한 쿨다운 기준 시점
    cooldown_from_index = None

    while True:
        try:
            candles = client.fetch_ohlcv(
                config.SYMBOL,
                timeframe=config.TIMEFRAME,
                limit=config.LIMIT,
            )
            if len(candles) < 2:
                print("Not enough candles yet")
                time.sleep(config.POLL_SECONDS)
                continue

            last_price = candles[-1][4]
            current_index = candles[-1][0]

            # 1) 포지션 보유 중이면 신규 진입 신호를 무시하고, TP/SL/타임스탑만 먼저 관리합니다.
            if in_position:
                candles_open = _count_candles_since(candles, entry_index)
                exit_reason = None

                if side == "long":
                    tp_price = entry_price * (1 + config.TAKE_PROFIT_PCT)
                    sl_price = entry_price * (1 - config.STOP_LOSS_PCT)
                    if last_price >= tp_price:
                        exit_reason = "take profit"
                    elif last_price <= sl_price:
                        exit_reason = "stop loss"
                else:  # short
                    tp_price = entry_price * (1 - config.TAKE_PROFIT_PCT)
                    sl_price = entry_price * (1 + config.STOP_LOSS_PCT)
                    if last_price <= tp_price:
                        exit_reason = "take profit"
                    elif last_price >= sl_price:
                        exit_reason = "stop loss"

                # 타임스탑: TP/SL 미도달 상태로 12개 캔들이 지나면 시장가 청산
                if exit_reason is None and candles_open >= config.TIME_STOP_CANDLES:
                    exit_reason = "time stop"

                print(
                    f"[POSITION] side={side} entry={entry_price} last={last_price} candles_open={candles_open}"
                )

                if exit_reason:
                    exit_side = "sell" if side == "long" else "buy"

                    # 페이퍼 트레이딩: 실제 주문 없이 로그만 출력
                    if config.LIVE_TRADING:
                        order = client.create_market_order(
                            config.SYMBOL,
                            exit_side,
                            config.ORDER_SIZE,
                        )
                        print(f"[LIVE] Exit {side} ({exit_reason}):", order)
                    else:
                        print(
                            f"[PAPER] Would place market {exit_side.upper()} for {side} exit ({exit_reason})"
                        )

                    # 손절이면 쿨다운 시작점 저장 (이후 8캔들 동안 신규 진입 금지)
                    if exit_reason == "stop loss":
                        cooldown_from_index = current_index

                    in_position = False
                    side = None
                    entry_price = None
                    entry_index = None

            # 2) 포지션이 없을 때만 신규 진입 평가 (쿨다운 중이면 hold)
            if not in_position:
                if cooldown_from_index is not None:
                    cooldown_passed = _count_candles_since(candles, cooldown_from_index)
                    if cooldown_passed < config.COOLDOWN_AFTER_SL_CANDLES:
                        print(
                            f"[COOLDOWN] hold ({cooldown_passed}/{config.COOLDOWN_AFTER_SL_CANDLES} candles)"
                        )
                        time.sleep(config.POLL_SECONDS)
                        continue

                signal, rsi, adx, reason = generate_signal(candles, config)
                print(
                    f"[SIGNAL] price={last_price} signal={signal} rsi={rsi} adx={adx} reason={reason}"
                )

                if signal in {"buy", "sell"}:
                    order_side = "buy" if signal == "buy" else "sell"
                    position_side = "long" if signal == "buy" else "short"

                    if config.LIVE_TRADING:
                        order = client.create_market_order(
                            config.SYMBOL,
                            order_side,
                            config.ORDER_SIZE,
                        )
                        print(f"[LIVE] Entry {position_side}:", order)
                    else:
                        print(
                            f"[PAPER] Would place market {order_side.upper()} for {position_side} entry"
                        )

                    in_position = True
                    side = position_side
                    # 현재는 단순화를 위해 최신 종가를 진입가로 사용합니다.
                    # 실제 라이브 환경에서는 체결가(fill price)를 저장해야 더 정확합니다.
                    entry_price = last_price
                    entry_index = current_index

            time.sleep(config.POLL_SECONDS)
        except Exception as exc:
            print("Bot error:", exc)
            time.sleep(config.POLL_SECONDS)


if __name__ == "__main__":
    run_bot()
