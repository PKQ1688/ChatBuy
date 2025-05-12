import pandas as pd
import vectorbt as vbt

# 读取数据
df = pd.read_csv(
    "data/BTC_USDT_1d_with_indicators.csv", index_col=0, parse_dates=True
)
df = df.dropna(subset=["Close", "bb_upper", "bb_middle", "bb_lower", "rsi"])

close = df["Close"]
bb_upper = df["bb_upper"]
bb_lower = df["bb_lower"]
rsi = df["rsi"]

# 入场信号: 收盘价上穿下轨 且 RSI上穿30
entries = (close.vbt.crossed_above(bb_lower)) & (rsi.vbt.crossed_above(30))
# 离场信号: 收盘价下穿上轨 且 RSI下穿70
exits = (close.vbt.crossed_below(bb_upper)) & (rsi.vbt.crossed_below(70))

# 回测
pf = vbt.Portfolio.from_signals(close, entries, exits, fees=0.002, init_cash=1_000_000)

print(pf.stats())