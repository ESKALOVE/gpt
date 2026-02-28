"""Minimal backtesting module for the live bot strategy."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd

import config
from exchange import GateioFuturesClient
from strategy import generate_signal


def timeframe_to_ms(timeframe):
    """ccxt timeframe 문자열(예: 15m)을 ms로 변환합니다."""
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    unit_ms = {
        "m": 60_000,
        "h": 3_600_000,
        "d": 86_400_000,
        "w": 604_800_000,
    }
    return value * unit_ms.get(unit, 1)


def _format_ts(ts_ms):
    if ts_ms is None:
        return "None"
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()


def fetch_ohlcv_paged(
    client,
    symbol,
    timeframe,
    total_candles,
    chunk_limit,
    since=None,
    debug_fetch=False,
):
    """since 기반 페이지 조회로 단일 limit보다 긴 히스토리를 수집합니다."""
    all_candles = []
    next_since = since
    timeframe_ms = timeframe_to_ms(timeframe)
    call_no = 0

    # 페이징 방식: 같은 심볼/타임프레임을 since를 앞으로 밀면서 반복 조회합니다.
    while len(all_candles) < total_candles:
        call_no += 1
        remain = total_candles - len(all_candles)
        fetch_limit = min(chunk_limit, remain)

        batch = client.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            limit=fetch_limit,
            since=next_since,
        )

        if debug_fetch:
            print(
                f"[FETCH] call={call_no} since_ms={next_since} since_utc={_format_ts(next_since)} returned={len(batch)}"
            )

        if not batch:
            break

        all_candles.extend(batch)

        last_ts = batch[-1][0]
        # 다음 조회 시작 시점은 마지막 캔들 이후로 이동합니다.
        # 기본은 timeframe_ms를 더하고, 비정상 값이면 +1ms를 사용합니다.
        next_since = last_ts + timeframe_ms if timeframe_ms > 0 else last_ts + 1

    # timestamp 기준 dedupe + 정렬 (거래소 응답 중복 대비)
    by_ts = {row[0]: row for row in all_candles}
    ordered = [by_ts[ts] for ts in sorted(by_ts.keys())]
    return ordered[:total_candles]


def build_ohlcv_df(
    client,
    symbol,
    timeframe,
    total_candles,
    chunk_limit,
    since=None,
    debug_fetch=False,
):
    candles = fetch_ohlcv_paged(
        client=client,
        symbol=symbol,
        timeframe=timeframe,
        total_candles=total_candles,
        chunk_limit=chunk_limit,
        since=since,
        debug_fetch=debug_fetch,
    )
    df = pd.DataFrame(
        candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    return df


def apply_slippage(price, side, is_entry, slippage_pct):
    """보수적으로(불리하게) 슬리피지를 진입/청산 체결가에 반영합니다."""
    if slippage_pct <= 0:
        return price

    if side == "long":
        return price * (1 + slippage_pct) if is_entry else price * (1 - slippage_pct)

    return price * (1 - slippage_pct) if is_entry else price * (1 + slippage_pct)


def calc_trade_return(side, entry_price, exit_price, fee_rate):
    """거래 1건 수익률(수수료 포함)을 계산합니다."""
    if side == "long":
        gross = (exit_price / entry_price) - 1
    else:
        gross = (entry_price / exit_price) - 1
    return gross - 2 * fee_rate


def print_signal_debug_counts(df, cfg):
    # debug 출력 목적: 실제 진입 이전에 신호 분포/hold 사유를 확인해 전략이 너무 엄격한지 점검합니다.
    candles = df[["timestamp", "open", "high", "low", "close", "volume"]].values.tolist()
    min_needed = max(cfg.BB_PERIOD + 1, cfg.RSI_PERIOD + 1, cfg.ADX_PERIOD + 1)

    signal_counts = Counter()
    reason_counts = Counter()

    for i in range(min_needed, len(candles)):
        signal, _, _, reason = generate_signal(candles[: i + 1], cfg)
        signal_counts[signal] += 1
        reason_counts[reason] += 1

    print(f"[DEBUG] signal counts: {dict(signal_counts)}")
    print("[DEBUG] top reasons:")
    for reason, count in reason_counts.most_common(10):
        print(f"  - {reason}: {count}")


def run_backtest(df, cfg):
    # 캔들 기반 체결 가정:
    # - 진입은 '신호가 발생한 캔들 종가'로 체결.
    # - TP/SL은 항상 '다음 캔들(high/low)'로 터치 여부를 판정.
    # - 같은 캔들에서 TP/SL 동시 충족 시 보수적으로 SL 우선 처리.

    candles = df[["timestamp", "open", "high", "low", "close", "volume"]].values.tolist()
    min_needed = max(cfg.BB_PERIOD + 1, cfg.RSI_PERIOD + 1, cfg.ADX_PERIOD + 1)

    in_position = False
    side = None
    entry_price = None
    entry_idx = None
    entry_time = None

    # 쿨다운 처리: 손절 발생 후 지정 캔들 수만큼 신규 진입 차단.
    cooldown_until_idx = -1
    trades = []

    for i in range(min_needed, len(candles) - 1):
        curr = candles[i]
        nxt = candles[i + 1]

        if in_position:
            next_high = nxt[2]
            next_low = nxt[3]
            next_close = nxt[4]
            exit_price_raw = None
            exit_reason = None

            if side == "long":
                tp_price = entry_price * (1 + cfg.TAKE_PROFIT_PCT)
                sl_price = entry_price * (1 - cfg.STOP_LOSS_PCT)

                # TP/SL 판정은 다음 캔들 high/low를 사용하고, 충돌 시 SL 우선.
                if next_low <= sl_price:
                    exit_price_raw = sl_price
                    exit_reason = "stop loss"
                elif next_high >= tp_price:
                    exit_price_raw = tp_price
                    exit_reason = "take profit"
            else:
                tp_price = entry_price * (1 - cfg.TAKE_PROFIT_PCT)
                sl_price = entry_price * (1 + cfg.STOP_LOSS_PCT)

                if next_high >= sl_price:
                    exit_price_raw = sl_price
                    exit_reason = "stop loss"
                elif next_low <= tp_price:
                    exit_price_raw = tp_price
                    exit_reason = "take profit"

            # 타임스탑: 진입 이후 12캔들 경과 시 다음 캔들 종가로 청산.
            candles_open = (i + 1) - entry_idx
            if exit_reason is None and candles_open >= cfg.TIME_STOP_CANDLES:
                exit_price_raw = next_close
                exit_reason = "time stop"

            if exit_reason is not None:
                exit_price = apply_slippage(
                    exit_price_raw,
                    side=side,
                    is_entry=False,
                    slippage_pct=cfg.SLIPPAGE_PCT,
                )
                pnl = calc_trade_return(side, entry_price, exit_price, cfg.FEE_RATE)
                trades.append(
                    {
                        "entry_time": pd.to_datetime(entry_time, unit="ms"),
                        "exit_time": pd.to_datetime(nxt[0], unit="ms"),
                        "side": side,
                        "entry": entry_price,
                        "exit": exit_price,
                        "reason": exit_reason,
                        "pnl_pct": pnl * 100,
                    }
                )

                if exit_reason == "stop loss":
                    cooldown_until_idx = (i + 1) + cfg.COOLDOWN_AFTER_SL_CANDLES

                in_position = False
                side = None
                entry_price = None
                entry_idx = None
                entry_time = None

            continue

        if i < cooldown_until_idx:
            continue

        signal, rsi, adx, reason = generate_signal(candles[: i + 1], cfg)
        _ = (rsi, adx, reason)
        if signal not in {"buy", "sell"}:
            continue

        side = "long" if signal == "buy" else "short"
        entry_raw = curr[4]
        entry_price = apply_slippage(
            entry_raw,
            side=side,
            is_entry=True,
            slippage_pct=cfg.SLIPPAGE_PCT,
        )
        entry_idx = i
        entry_time = curr[0]
        in_position = True

    if in_position:
        last = candles[-1]
        exit_price = apply_slippage(
            last[4],
            side=side,
            is_entry=False,
            slippage_pct=cfg.SLIPPAGE_PCT,
        )
        pnl = calc_trade_return(side, entry_price, exit_price, cfg.FEE_RATE)
        trades.append(
            {
                "entry_time": pd.to_datetime(entry_time, unit="ms"),
                "exit_time": pd.to_datetime(last[0], unit="ms"),
                "side": side,
                "entry": entry_price,
                "exit": exit_price,
                "reason": "end of data",
                "pnl_pct": pnl * 100,
            }
        )

    return pd.DataFrame(trades)


def print_summary(trades_df):
    if trades_df.empty:
        print("No trades found.")
        return

    returns = trades_df["pnl_pct"] / 100
    equity = (1 + returns).cumprod()
    equity_curve = pd.concat([pd.Series([1.0]), equity], ignore_index=True)
    peak = equity_curve.cummax()
    drawdown = (equity_curve / peak - 1) * 100

    wins = trades_df[trades_df["pnl_pct"] > 0]
    losses = trades_df[trades_df["pnl_pct"] < 0]
    profit_factor = (
        wins["pnl_pct"].sum() / abs(losses["pnl_pct"].sum()) if not losses.empty else float("inf")
    )

    total_return = (equity_curve.iloc[-1] - 1) * 100
    winrate = (len(wins) / len(trades_df)) * 100
    avg_trade = trades_df["pnl_pct"].mean()
    max_dd = drawdown.min()

    print("=== Backtest Summary ===")
    print(f"Trades: {len(trades_df)}")
    print(f"Winrate: {winrate:.2f}%")
    print(f"Total return: {total_return:.2f}%")
    print(f"Max drawdown: {max_dd:.2f}%")
    print(f"Avg trade: {avg_trade:.3f}%")
    print(f"Profit factor: {profit_factor:.3f}" if profit_factor != float("inf") else "Profit factor: inf")

    print("\n=== Last 10 Trades ===")
    cols = ["entry_time", "side", "entry", "exit", "reason", "pnl_pct"]
    print(trades_df[cols].tail(10).to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description="Simple backtest for Gate.io futures strategy")
    parser.add_argument("--symbol", default=config.SYMBOL)
    parser.add_argument("--timeframe", default=config.TIMEFRAME)
    parser.add_argument("--total_candles", type=int, default=10000)
    parser.add_argument("--chunk_limit", type=int, default=2000)
    parser.add_argument("--since", type=int, default=None)
    parser.add_argument("--preset", choices=["strict", "relaxed"], default="strict")
    parser.add_argument("--fee-rate", type=float, default=config.FEE_RATE)
    parser.add_argument("--slippage-pct", type=float, default=config.SLIPPAGE_PCT)
    parser.add_argument("--debug_fetch", action="store_true")
    parser.add_argument("--save-csv", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = SimpleNamespace(**{k: getattr(config, k) for k in dir(config) if k.isupper()})
    cfg.TIMEFRAME = args.timeframe
    cfg.FEE_RATE = args.fee_rate
    cfg.SLIPPAGE_PCT = args.slippage_pct

    # strict/relaxed 프리셋은 백테스트에서만 신호 민감도 점검용으로 사용합니다(실거래 설정은 변경하지 않음).
    if args.preset == "relaxed":
        cfg.RSI_OVERSOLD = 40
        cfg.RSI_OVERBOUGHT = 60
        cfg.ADX_MAX = 35

    print(
        f"Preset={args.preset} | RSI_OVERSOLD={cfg.RSI_OVERSOLD} RSI_OVERBOUGHT={cfg.RSI_OVERBOUGHT} ADX_MAX={cfg.ADX_MAX}"
    )

    client = GateioFuturesClient(config.API_KEY, config.API_SECRET)
    df = build_ohlcv_df(
        client=client,
        symbol=args.symbol,
        timeframe=args.timeframe,
        total_candles=args.total_candles,
        chunk_limit=args.chunk_limit,
        since=args.since,
        debug_fetch=args.debug_fetch,
    )

    print(
        f"Requested candles={args.total_candles}, loaded candles={len(df)} | symbol={args.symbol} timeframe={args.timeframe}"
    )

    print_signal_debug_counts(df, cfg)
    trades_df = run_backtest(df, cfg)
    print_summary(trades_df)

    if args.save_csv and not trades_df.empty:
        trades_df.to_csv("trades.csv", index=False)
        print("Saved trades to trades.csv")


if __name__ == "__main__":
    main()
