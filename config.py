"""Configuration for the Gate.io futures trading bot."""

API_KEY = "YOUR_GATEIO_API_KEY"
API_SECRET = "YOUR_GATEIO_API_SECRET"

SYMBOL = "BTC/USDT:USDT"
TIMEFRAME = "15m"
LIMIT = 100

# RSI(14): 과매도/과매수 구간을 진입 후보로 사용합니다.
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# BB(20, 2): 밴드 밖으로 이탈했다가 다시 밴드 안으로 복귀(re-entry)할 때만 진입합니다.
BB_PERIOD = 20
BB_STD = 2.0

# ADX(14): 추세 강도가 너무 강하면(>25) 역추세 진입을 피하기 위해 필터링합니다.
ADX_PERIOD = 14
ADX_MAX = 25

# 손익절 기준: 진입가 대비 +0.9% 이익실현, -0.7% 손절.
TAKE_PROFIT_PCT = 0.009
STOP_LOSS_PCT = 0.007

# 타임스탑: 12개 캔들 동안 TP/SL 미도달이면 시장가 청산.
TIME_STOP_CANDLES = 12

# 손절 직후 8개 캔들 동안 신규 진입 금지(쿨다운).
COOLDOWN_AFTER_SL_CANDLES = 8

ORDER_SIZE = 0.001
POLL_SECONDS = 30

# False면 페이퍼 트레이딩(주문 미실행, 로그만 출력), True면 실제 주문을 전송합니다.
LIVE_TRADING = False
