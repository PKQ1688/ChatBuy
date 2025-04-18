import pandas as pd

# 读取信号数据
signals = pd.read_csv("output/trade_advice_unified_results_one.csv")
signals = signals.rename(columns={"trade_time": "timestamp"})

# 读取价格数据
prices = pd.read_csv("data/BTC_USDT_1d_with_indicators.csv")

# 合并信号和价格
df = pd.merge(signals, prices[["timestamp", "close"]], on="timestamp", how="inner")

# 模拟交易
position = 0
buy_price = 0
trades = []
trade_dates = []
for idx, row in df.iterrows():
    if row["action"] == "buy" and position == 0:
        buy_price = row["close"]
        position = 1
        trade_dates.append((row["timestamp"], "buy", buy_price))
    elif row["action"] == "sell" and position == 1:
        sell_price = row["close"]
        trades.append(sell_price - buy_price)
        trade_dates.append((row["timestamp"], "sell", sell_price))
        position = 0

# 如果最后还有持仓，强制在最后一根K线卖出
if position == 1 and not df.empty:
    last_row = df.iloc[-1]
    sell_price = last_row["close"]
    trades.append(sell_price - buy_price)
    trade_dates.append((last_row["timestamp"], "sell", sell_price))
    position = 0

# 统计结果
total_profit = sum(trades)
win_count = sum([1 for t in trades if t > 0])
win_rate = win_count / len(trades) if trades else 0
avg_profit = total_profit / len(trades) if trades else 0

print(f"总交易次数: {len(trades)}")
print(f"总收益: {total_profit:.2f}")
print(f"胜率: {win_rate:.2%}")
print(f"平均单笔收益: {avg_profit:.2f}")
print("每笔交易明细:")
for d in trade_dates:
    print(d)
