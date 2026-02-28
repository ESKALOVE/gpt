"""Indicator calculation and entry signal generation."""


def _sma(values, period):
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def calculate_rsi(closes, period=14):
    if len(closes) <= period:
        return None

    gains = []
    losses = []
    for i in range(1, len(closes)):
        change = closes[i] - closes[i - 1]
        gains.append(max(change, 0))
        losses.append(abs(min(change, 0)))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_bollinger_bands(closes, period=20, std_mult=2.0):
    if len(closes) < period:
        return None

    window = closes[-period:]
    mid = sum(window) / period
    variance = sum((x - mid) ** 2 for x in window) / period
    std = variance**0.5
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return mid, upper, lower


def calculate_adx(highs, lows, closes, period=14):
    if len(closes) <= period:
        return None

    tr_list = []
    plus_dm_list = []
    minus_dm_list = []

    for i in range(1, len(closes)):
        high_diff = highs[i] - highs[i - 1]
        low_diff = lows[i - 1] - lows[i]

        plus_dm = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm = low_diff if low_diff > high_diff and low_diff > 0 else 0

        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )

        tr_list.append(tr)
        plus_dm_list.append(plus_dm)
        minus_dm_list.append(minus_dm)

    if len(tr_list) < period:
        return None

    atr = sum(tr_list[:period]) / period
    plus_dm_smoothed = sum(plus_dm_list[:period]) / period
    minus_dm_smoothed = sum(minus_dm_list[:period]) / period

    dx_values = []
    for i in range(period, len(tr_list)):
        atr = (atr * (period - 1) + tr_list[i]) / period
        plus_dm_smoothed = (plus_dm_smoothed * (period - 1) + plus_dm_list[i]) / period
        minus_dm_smoothed = (minus_dm_smoothed * (period - 1) + minus_dm_list[i]) / period

        if atr == 0:
            continue

        plus_di = (plus_dm_smoothed / atr) * 100
        minus_di = (minus_dm_smoothed / atr) * 100
        denominator = plus_di + minus_di
        dx = 0 if denominator == 0 else abs(plus_di - minus_di) / denominator * 100
        dx_values.append(dx)

    if not dx_values:
        return None

    return _sma(dx_values, min(period, len(dx_values)))


def generate_signal(candles, cfg):
    """캔들(OHLCV) 기반으로 buy/sell/hold 신호를 생성합니다."""
    if len(candles) < max(cfg.BB_PERIOD + 1, cfg.RSI_PERIOD + 1, cfg.ADX_PERIOD + 1):
        return "hold", None, None, "not enough candles"

    closes = [c[4] for c in candles]
    highs = [c[2] for c in candles]
    lows = [c[3] for c in candles]

    rsi = calculate_rsi(closes, period=cfg.RSI_PERIOD)
    adx = calculate_adx(highs, lows, closes, period=cfg.ADX_PERIOD)

    if rsi is None or adx is None:
        return "hold", rsi, adx, "indicator not ready"

    # ADX가 높으면 추세가 강한 구간이므로, 역추세 성격의 BB 재진입 전략 진입을 차단합니다.
    if adx > cfg.ADX_MAX:
        return "hold", rsi, adx, "adx too high"

    prev_closes = closes[:-1]
    curr_closes = closes
    prev_bb = calculate_bollinger_bands(prev_closes, cfg.BB_PERIOD, cfg.BB_STD)
    curr_bb = calculate_bollinger_bands(curr_closes, cfg.BB_PERIOD, cfg.BB_STD)

    if prev_bb is None or curr_bb is None:
        return "hold", rsi, adx, "bb not ready"

    _, prev_upper, prev_lower = prev_bb
    _, curr_upper, curr_lower = curr_bb

    prev_close = closes[-2]
    curr_close = closes[-1]

    # BB 재진입 의미: 이전 봉은 밴드 밖, 현재 봉은 다시 밴드 안으로 복귀한 상태를 뜻합니다.
    long_reentry = prev_close < prev_lower and curr_close >= curr_lower
    short_reentry = prev_close > prev_upper and curr_close <= curr_upper

    if rsi < cfg.RSI_OVERSOLD and long_reentry:
        return "buy", rsi, adx, "long: rsi oversold + bb re-entry"

    if rsi > cfg.RSI_OVERBOUGHT and short_reentry:
        return "sell", rsi, adx, "short: rsi overbought + bb re-entry"

    return "hold", rsi, adx, "conditions not met"
