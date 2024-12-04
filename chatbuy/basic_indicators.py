import pandas as pd

from talipp.indicators import MACD, BB

data = pd.read_csv("data/BTC_USDT_1d.csv")


# 计算 MACD 指标
macd = MACD(12, 26, 9, data["close"])[:]
bb = BB(20, 2, data["close"])[:]

for index, row in data.iterrows():
    if macd[index] is not None:
        data.at[index, "macd"] = macd[index].macd
        data.at[index, "signal"] = macd[index].signal
        data.at[index, "histogram"] = macd[index].histogram

    if bb[index] is not None:
        data.at[index, "bb_upper"] = bb[index].ub
        data.at[index, "bb_middle"] = bb[index].cb
        data.at[index, "bb_lower"] = bb[index].lb

data = data.infer_objects(copy=False)
data.fillna(0, inplace=True)

data.to_csv("data/BTC_USDT_1d_with_indicators.csv", index=False)
