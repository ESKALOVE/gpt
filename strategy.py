"""Simple RSI strategy example."""


def calculate_rsi(closes, period: int = 14):
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

    if avg_loss == 0:
        return 100

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def generate_signal(rsi_value: float, oversold: float = 30, overbought: float = 70):
    if rsi_value is None:
        return "hold"
    if rsi_value < oversold:
        return "buy"
    if rsi_value > overbought:
        return "sell"
    return "hold"
