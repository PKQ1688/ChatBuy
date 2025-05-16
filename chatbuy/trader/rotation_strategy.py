import vectorbt as vbt
import numpy as np
import pandas as pd


def run_rotation_strategy(price_df: pd.DataFrame, window: int = 20):
    """
    轮动策略：每个周期买入过去window日涨幅最大的币种（BTC或ETH）。
    :param price_df: 行为日期，列为币种（如BTC、ETH）的收盘价DataFrame
    :param window: 计算涨幅的窗口长度
    :return: vectorbt Portfolio对象
    """
    # 计算过去window日的涨幅
    returns = price_df.pct_change(window)
    # 每天选择涨幅最大的币种
    best = returns.idxmax(axis=1)
    # 生成进出场信号
    entries = pd.DataFrame(False, index=price_df.index, columns=price_df.columns)
    exits = pd.DataFrame(False, index=price_df.index, columns=price_df.columns)
    for i in range(window, len(price_df)):
        coin = best.iloc[i]
        entries.iloc[i, price_df.columns.get_loc(coin)] = True
        # 其余币种全部退出
        for c in price_df.columns:
            if c != coin:
                exits.iloc[i, price_df.columns.get_loc(c)] = True
    # 用vectorbt生成组合
    pf = vbt.Portfolio.from_signals(price_df, entries=entries, exits=exits, freq="1D")
    return pf


# 示例用法（假设已准备好price_df）
price_df = vbt.YFData.download(["BTC-USD", "ETH-USD"], period="1y").get("Close")
pf = run_rotation_strategy(price_df)
print(pf.stats())
