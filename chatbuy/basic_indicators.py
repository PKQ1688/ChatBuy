import pandas as pd

from talipp.indicators import SMA, MACD, BB

data = pd.read_csv("data/BTC_USDT_1d.csv")

# 计算 20 日简单移动平均线
sma = SMA(20, data["close"])
data["sma"] = sma

# 计算 MACD 指标
macd = MACD(12, 26, 9, data["close"])[-1]
data["macd"] = macd.macd
data["signal"] = macd.signal
data["histogram"] = macd.histogram

# print(data.head())

bb = BB(20, 2, data["close"])[-1]
data["bb_upper"] = bb.ub
data["bb_middle"] = bb.cb
data["bb_lower"] = bb.lb

data.to_csv("data/BTC_USDT_1d_with_indicators.csv", index=False)
