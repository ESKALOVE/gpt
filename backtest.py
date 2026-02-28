"""Minimal backtesting module for the live bot strategy."""

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
from types import SimpleNamespace

import config
from exchange import GateioFuturesClient
from strategy import generate_signal


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
    """Gate.io 선물은 from/to 윈도우 조회를 사용해 안전하게 페이징합니다."""
    exchange = client.exchange
    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000

    # Gate.io 선물 제약: from/to와 limit를 같이 보내면 오류가 나므로,
    # limit 없이 from/to 시간 구간만으로 조회합니다.
    end_ms = exchange.milliseconds()
    start_ms = since if since is not None else end_ms - total_candles * timeframe_ms

    all_candles = []
    window_start_ms = start_ms
    call_no = 0
    stale_rounds = 0

    while window_start_ms < end_ms and len(all_candles) < total_candles:
        call_no += 1
        # chunk_limit은 요청 limit가 아니라, 한 번에 조회할 시간 윈도우 크기입니다.
        window_end_ms = min(window_start_ms + chunk_limit * timeframe_ms, end_ms)

        batch = client.exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            params={
                "from": int(window_start_ms / 1000),
                "to": int(window_end_ms / 1000),
            },
        )

        if debug_fetch:
            window_size_candles = max(1, int((window_end_ms - window_start_ms) / timeframe_ms))
            print(
                "[FETCH] "
                f"call={call_no} "
                "style=from_to_only(limit_not_sent=True) "
                f"window_start_ms={int(window_start_ms)} "
                f"window_start_utc={_format_ts(window_start_ms)} "
                f"window_end_ms={int(window_end_ms)} "
                f"window_end_utc={_format_ts(window_end_ms)} "
                f"window_size_candles={window_size_candles} "
                f"returned={len(batch)}"
            )

        if not batch:
            stale_rounds += 1
            if stale_rounds >= 3:
                break
            window_start_ms = window_end_ms
            continue

        before = len(all_candles)
        all_candles.extend(batch)

        # 거래소가 from/to를 무시하고 같은 데이터를 반복 반환할 수 있어
        # 진전이 없는 상태가 반복되면 안전하게 중단합니다.
        last_ts = batch[-1][0]
        proposed_next = max(window_end_ms, last_ts + timeframe_ms)
        if proposed_next <= window_start_ms or len(all_candles) == before:
            stale_rounds += 1
            if stale_rounds >= 3:
                break
            window_start_ms = window_end_ms
        else:
            stale_rounds = 0
            window_start_ms = proposed_next

    by_ts = {row[0]: row for row in all_candles}
    ordered = [by_ts[ts] for ts in sorted(by_ts.keys())]
    return ordered[-total_candles:]


def build_ohlcv_rows(client, symbol, timeframe, total_candles, chunk_limit, since=None, debug_fetch=False):
    return fetch_ohlcv_paged(
        client=client,
        symbol=symbol,
        timeframe=timeframe,
        total_candles=total_candles,
        chunk_limit=chunk_limit,
        since=since,
        debug_fetch=debug_fetch,
    )


def apply_slippage(price, side, is_entry, slippage_pct):
    if slippage_pct <= 0:
        return price
    if side == "long":
        return price * (1 + slippage_pct) if is_entry else price * (1 - slippage_pct)
    return price * (1 - slippage_pct) if is_entry else price * (1 + slippage_pct)


def calc_trade_return(side, entry_price, exit_price, fee_rate):
    gross = (exit_price / entry_price) - 1 if side == "long" else (entry_price / exit_price) - 1
    return gross - 2 * fee_rate


def print_signal_debug_counts(candles, cfg):
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


def run_backtest(candles, cfg):
    min_needed = max(cfg.BB_PERIOD + 1, cfg.RSI_PERIOD + 1, cfg.ADX_PERIOD + 1)

    in_position = False
    side = None
    entry_price = None
    entry_idx = None
    entry_time = None

    cooldown_until_idx = -1
    trades = []

    for i in range(min_needed, len(candles) - 1):
        curr = candles[i]
        nxt = candles[i + 1]

        if in_position:
            next_high, next_low, next_close = nxt[2], nxt[3], nxt[4]
            exit_price_raw = None
            exit_reason = None

            if side == "long":
                tp_price = entry_price * (1 + cfg.TAKE_PROFIT_PCT)
                sl_price = entry_price * (1 - cfg.STOP_LOSS_PCT)
                if next_low <= sl_price:
                    exit_price_raw, exit_reason = sl_price, "stop loss"
                elif next_high >= tp_price:
                    exit_price_raw, exit_reason = tp_price, "take profit"
            else:
                tp_price = entry_price * (1 - cfg.TAKE_PROFIT_PCT)
                sl_price = entry_price * (1 + cfg.STOP_LOSS_PCT)
                if next_high >= sl_price:
                    exit_price_raw, exit_reason = sl_price, "stop loss"
                elif next_low <= tp_price:
                    exit_price_raw, exit_reason = tp_price, "take profit"

            candles_open = (i + 1) - entry_idx
            if exit_reason is None and candles_open >= cfg.TIME_STOP_CANDLES:
                exit_price_raw, exit_reason = next_close, "time stop"

            if exit_reason is not None:
                exit_price = apply_slippage(exit_price_raw, side=side, is_entry=False, slippage_pct=cfg.SLIPPAGE_PCT)
                pnl = calc_trade_return(side, entry_price, exit_price, cfg.FEE_RATE)
                trades.append(
                    {
                        "entry_time": datetime.fromtimestamp(entry_time / 1000, tz=timezone.utc).isoformat(),
                        "exit_time": datetime.fromtimestamp(nxt[0] / 1000, tz=timezone.utc).isoformat(),
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

        signal, _, _, _ = generate_signal(candles[: i + 1], cfg)
        if signal not in {"buy", "sell"}:
            continue

        side = "long" if signal == "buy" else "short"
        entry_price = apply_slippage(curr[4], side=side, is_entry=True, slippage_pct=cfg.SLIPPAGE_PCT)
        entry_idx = i
        entry_time = curr[0]
        in_position = True

    if in_position:
        last = candles[-1]
        exit_price = apply_slippage(last[4], side=side, is_entry=False, slippage_pct=cfg.SLIPPAGE_PCT)
        pnl = calc_trade_return(side, entry_price, exit_price, cfg.FEE_RATE)
        trades.append(
            {
                "entry_time": datetime.fromtimestamp(entry_time / 1000, tz=timezone.utc).isoformat(),
                "exit_time": datetime.fromtimestamp(last[0] / 1000, tz=timezone.utc).isoformat(),
                "side": side,
                "entry": entry_price,
                "exit": exit_price,
                "reason": "end of data",
                "pnl_pct": pnl * 100,
            }
        )

    return trades


def print_summary(trades):
    if not trades:
        print("No trades found.")
        return

    returns = [t["pnl_pct"] / 100 for t in trades]
    equity = [1.0]
    for r in returns:
        equity.append(equity[-1] * (1 + r))

    peaks = []
    peak = equity[0]
    for e in equity:
        peak = max(peak, e)
        peaks.append(peak)
    drawdowns = [((e / p) - 1) * 100 for e, p in zip(equity, peaks)]

    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] < 0]
    win_sum = sum(t["pnl_pct"] for t in wins)
    loss_sum = sum(t["pnl_pct"] for t in losses)
    profit_factor = (win_sum / abs(loss_sum)) if loss_sum != 0 else float("inf")

    total_return = (equity[-1] - 1) * 100
    winrate = len(wins) / len(trades) * 100
    avg_trade = sum(t["pnl_pct"] for t in trades) / len(trades)
    max_dd = min(drawdowns)

    print("=== Backtest Summary ===")
    print(f"Trades: {len(trades)}")
    print(f"Winrate: {winrate:.2f}%")
    print(f"Total return: {total_return:.2f}%")
    print(f"Max drawdown: {max_dd:.2f}%")
    print(f"Avg trade: {avg_trade:.3f}%")
    print(f"Profit factor: {profit_factor:.3f}" if profit_factor != float("inf") else "Profit factor: inf")

    print("\n=== Last 10 Trades ===")
    for row in trades[-10:]:
        print(
            f"{row['entry_time']} | {row['side']} | entry={row['entry']:.6f} | "
            f"exit={row['exit']:.6f} | {row['reason']} | pnl={row['pnl_pct']:.3f}%"
        )


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

    if args.preset == "relaxed":
        cfg.RSI_OVERSOLD = 40
        cfg.RSI_OVERBOUGHT = 60
        cfg.ADX_MAX = 35

    print(
        f"Preset={args.preset} | RSI_OVERSOLD={cfg.RSI_OVERSOLD} RSI_OVERBOUGHT={cfg.RSI_OVERBOUGHT} ADX_MAX={cfg.ADX_MAX}"
    )

    client = GateioFuturesClient(config.API_KEY, config.API_SECRET)
    candles = build_ohlcv_rows(
        client=client,
        symbol=args.symbol,
        timeframe=args.timeframe,
        total_candles=args.total_candles,
        chunk_limit=args.chunk_limit,
        since=args.since,
        debug_fetch=args.debug_fetch,
    )

    print(
        f"Requested candles={args.total_candles}, loaded candles={len(candles)} | symbol={args.symbol} timeframe={args.timeframe}"
    )

    print_signal_debug_counts(candles, cfg)
    trades = run_backtest(candles, cfg)
    print_summary(trades)

    if args.save_csv and trades:
        with open("trades.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["entry_time", "exit_time", "side", "entry", "exit", "reason", "pnl_pct"],
            )
            writer.writeheader()
            writer.writerows(trades)
        print("Saved trades to trades.csv")


if __name__ == "__main__":
    main()
