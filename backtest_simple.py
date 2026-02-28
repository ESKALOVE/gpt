"""아주 단순한 Gate.io futures 백테스트 스크립트 (단일 fetch, 무페이징)."""

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
from types import SimpleNamespace

import config
from exchange import GateioFuturesClient
from strategy import generate_signal


def _ts_to_iso(ts_ms):
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()


def apply_slippage(price, side, is_entry, slippage_pct):
    """슬리피지를 보수적으로 반영합니다."""
    if slippage_pct <= 0:
        return price
    if side == "long":
        return price * (1 + slippage_pct) if is_entry else price * (1 - slippage_pct)
    return price * (1 - slippage_pct) if is_entry else price * (1 + slippage_pct)


def calc_trade_return(side, entry_price, exit_price, fee_rate):
    """왕복 수수료(2 * fee_rate) 포함 수익률 계산."""
    if side == "long":
        gross = (exit_price / entry_price) - 1
    else:
        gross = (entry_price / exit_price) - 1
    return gross - 2 * fee_rate


def print_signal_debug(candles, cfg):
    """신호 분포와 hold 사유를 빠르게 확인하기 위한 디버그 출력."""
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

    # 캔들 기반 체결 가정:
    # - 진입은 신호가 나온 현재 캔들 종가
    # - TP/SL은 다음 캔들 high/low로 판정
    # - TP/SL 동시 충족 시 보수적으로 SL 우선
    for i in range(min_needed, len(candles) - 1):
        curr = candles[i]
        nxt = candles[i + 1]

        if in_position:
            next_high = nxt[2]
            next_low = nxt[3]
            next_close = nxt[4]
            exit_reason = None
            exit_price_raw = None

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

            # 타임스탑: 포지션 오픈 후 지정 캔들 수 경과 시 다음 캔들 종가 청산
            candles_open = (i + 1) - entry_idx
            if exit_reason is None and candles_open >= cfg.TIME_STOP_CANDLES:
                exit_price_raw, exit_reason = next_close, "time stop"

            if exit_reason is not None:
                exit_price = apply_slippage(exit_price_raw, side, is_entry=False, slippage_pct=cfg.SLIPPAGE_PCT)
                pnl = calc_trade_return(side, entry_price, exit_price, cfg.FEE_RATE)
                trades.append(
                    {
                        "entry_time": _ts_to_iso(entry_time),
                        "exit_time": _ts_to_iso(nxt[0]),
                        "side": side,
                        "entry": entry_price,
                        "exit": exit_price,
                        "reason": exit_reason,
                        "pnl_pct": pnl * 100,
                    }
                )

                # 손절 이후 쿨다운: 일정 캔들 동안 신규 진입 차단
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
        entry_price = apply_slippage(curr[4], side, is_entry=True, slippage_pct=cfg.SLIPPAGE_PCT)
        entry_idx = i
        entry_time = curr[0]
        in_position = True

    if in_position:
        last = candles[-1]
        exit_price = apply_slippage(last[4], side, is_entry=False, slippage_pct=cfg.SLIPPAGE_PCT)
        pnl = calc_trade_return(side, entry_price, exit_price, cfg.FEE_RATE)
        trades.append(
            {
                "entry_time": _ts_to_iso(entry_time),
                "exit_time": _ts_to_iso(last[0]),
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

    peak = equity[0]
    drawdowns = []
    for e in equity:
        peak = max(peak, e)
        drawdowns.append(((e / peak) - 1) * 100)

    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] < 0]

    total_return = (equity[-1] - 1) * 100
    winrate = len(wins) / len(trades) * 100
    avg_trade = sum(t["pnl_pct"] for t in trades) / len(trades)
    max_dd = min(drawdowns)

    win_sum = sum(t["pnl_pct"] for t in wins)
    loss_sum = sum(t["pnl_pct"] for t in losses)
    profit_factor = (win_sum / abs(loss_sum)) if loss_sum != 0 else float("inf")

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
    parser = argparse.ArgumentParser(description="Minimal simple backtest without from/to paging")
    parser.add_argument("--symbol", default=config.SYMBOL)
    parser.add_argument("--timeframe", default=config.TIMEFRAME)
    parser.add_argument("--candles", type=int, default=1500)
    parser.add_argument("--preset", choices=["strict", "relaxed"], default="strict")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save-csv", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = SimpleNamespace(**{k: getattr(config, k) for k in dir(config) if k.isupper()})
    cfg.TIMEFRAME = args.timeframe

    # 테스트 전용 프리셋 완화
    if args.preset == "relaxed":
        cfg.RSI_OVERSOLD = 40
        cfg.RSI_OVERBOUGHT = 60
        cfg.ADX_MAX = 35

    client = GateioFuturesClient(config.API_KEY, config.API_SECRET)

    # Gate.io 선물 안전 구간: 단일 호출 limit를 1999 이하로 제한
    safe_limit = min(args.candles, 1999)
    candles = client.fetch_ohlcv(
        args.symbol,
        timeframe=args.timeframe,
        limit=safe_limit,
    )

    if not candles:
        print("No candles loaded.")
        return

    print(
        f"Loaded candles={len(candles)} | range={_ts_to_iso(candles[0][0])} -> {_ts_to_iso(candles[-1][0])}"
    )
    print(
        f"Preset={args.preset} | RSI_OVERSOLD={cfg.RSI_OVERSOLD} RSI_OVERBOUGHT={cfg.RSI_OVERBOUGHT} ADX_MAX={cfg.ADX_MAX}"
    )

    if args.debug:
        print_signal_debug(candles, cfg)

    trades = run_backtest(candles, cfg)
    print_summary(trades)

    if args.save_csv and trades:
        with open("trades_simple.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["entry_time", "exit_time", "side", "entry", "exit", "reason", "pnl_pct"],
            )
            writer.writeheader()
            writer.writerows(trades)
        print("Saved trades to trades_simple.csv")


if __name__ == "__main__":
    main()
