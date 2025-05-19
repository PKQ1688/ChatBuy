from collections import Counter

import akshare as ak
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vectorbt as vbt
from matplotlib.font_manager import FontProperties

# 设置中文字体支持
try:
    # 尝试设置中文字体，根据系统可用字体
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号
    
    # 尝试验证字体是否可用
    mpl.font_manager._rebuild()
    font_names = [f.name for f in mpl.font_manager.fontManager.ttflist]
    chinese_font = None
    
    for font in plt.rcParams["font.sans-serif"]:
        if font in font_names:
            chinese_font = FontProperties(fname=mpl.font_manager.findfont(font))
            print(f"使用中文字体: {font}")
            break
    
    if chinese_font is None:
        print("警告: 未找到合适的中文字体，图表中文可能显示为方块")
except Exception as e:
    print(f"设置中文字体时出错: {e}")
    print("图表中的中文可能显示不正常")

stock_zh_a_hist_df = ak.stock_zh_a_hist(
    symbol="399006",
    period="daily",
    start_date="20140101",
    end_date="20260101",
    adjust="hfq",
)

# 获取收盘价和每日涨跌幅
price = stock_zh_a_hist_df.get("收盘")
stock_zh_a_hist_df["daily_return"] = (
    stock_zh_a_hist_df["收盘"].pct_change() * 100
)  # 转换为百分比

# 计算年线 (MA250)
ma = vbt.MA.run(price, window=250, short_name="year")

# 计算MACD指标
macd = vbt.MACD.run(price, fast_window=12, slow_window=26, signal_window=9)

# 获取MACD柱状图数据 (histogram)
macd_histogram = macd.hist

# 创建条件
# 1. 价格在年线以上
price_above_ma = price > ma.ma

# 2. MACD柱状图为负
macd_hist_negative = macd_histogram < 0

# 3. 日跌幅超过1%（即日收益率小于-1%）
daily_decline_over_1pct = stock_zh_a_hist_df["daily_return"] < -1.0

# 统计每个条件的K线数量
count_above_ma = price_above_ma.sum()
count_negative_macd = macd_hist_negative.sum()
count_decline_1pct = daily_decline_over_1pct.sum()

# 合并所有条件
combined_condition = price_above_ma & macd_hist_negative & daily_decline_over_1pct
count_combined = combined_condition.sum()

# 输出结果
print(f"股价在年线以上的K线数量: {count_above_ma}")
print(f"MACD柱状为负的K线数量: {count_negative_macd}")
print(f"日跌幅超过1%的K线数量: {count_decline_1pct}")
print(f"满足所有条件的K线数量: {count_combined}")

# 尝试放宽条件，看看两个条件的组合情况
condition_1_2 = price_above_ma & macd_hist_negative
condition_1_3 = price_above_ma & daily_decline_over_1pct
condition_2_3 = macd_hist_negative & daily_decline_over_1pct

print(f"\n股价在年线以上且MACD柱状为负的K线数量: {condition_1_2.sum()}")
print(f"股价在年线以上且日跌幅超过1%的K线数量: {condition_1_3.sum()}")
print(f"MACD柱状为负且日跌幅超过1%的K线数量: {condition_2_3.sum()}")

# 显示部分满足条件的日期和数据（如果有）
if count_combined > 0:
    matching_days = stock_zh_a_hist_df[combined_condition]
    print("\n满足所有条件的交易日前10个:")
    if len(matching_days) > 10:
        print(matching_days)
        print(matching_days.iloc[-10:][["日期", "收盘", "daily_return"]])
    else:
        print(matching_days[["日期", "收盘", "daily_return"]])

    # # 计算统计数据
    # avg_decline = matching_days["daily_return"].mean()
    # max_decline = matching_days["daily_return"].min()
    # min_decline = matching_days["daily_return"].max()
    # std_decline = matching_days["daily_return"].std()

    # print(f"\n满足条件的日跌幅统计数据:")
    # print(f"总天数: {len(matching_days)}天")
    # print(f"平均跌幅: {avg_decline:.2f}%")
    # print(f"最大跌幅: {max_decline:.2f}%")
    # print(f"最小跌幅: {min_decline:.2f}%")
    # print(f"跌幅标准差: {std_decline:.2f}%")

    # # 分析这些天的次日表现
    # matching_days_indexes = matching_days.index
    # next_day_returns = []

    # for idx in matching_days_indexes:
    #     if idx + 1 < len(stock_zh_a_hist_df):
    #         next_day_return = stock_zh_a_hist_df.loc[idx + 1, "daily_return"]
    #         next_day_returns.append(next_day_return)

    # if next_day_returns:
    #     next_day_returns = pd.Series(next_day_returns)
    #     avg_next_day = next_day_returns.mean()
    #     positive_next_days = (next_day_returns > 0).sum()
    #     positive_rate = positive_next_days / len(next_day_returns) * 100

    #     print(f"\n次日表现统计:")
    #     print(f"次日平均收益率: {avg_next_day:.2f}%")
    #     print(
    #         f"次日上涨概率: {positive_rate:.2f}% ({positive_next_days}/{len(next_day_returns)})"
    #     )

# 统计每次MACD柱子连续为负的区间内，满足所有条件的K线平均数量
def analyze_consecutive_negative_macd():
    """分析每次MACD柱子连续为负期间满足条件的K线情况.

    该函数会计算每一次MACD柱子连续为负的区间内，
    满足所有条件（股价在年线以上 & MACD柱状为负 & 日跌幅超过1%）的K线平均数量。
    
    Returns:
        None: 函数直接打印结果，不返回值
    """
    print("\n\n分析每次MACD柱子连续为负期间满足条件的K线情况:")
    
    # 创建一个Series表示MACD柱子的状态（正/负）
    macd_status = macd_histogram.copy()
    
    # 创建一个负值标志的序列
    is_negative_series = macd_status < 0
    
    # 查找MACD状态变化点（从正变负或从负变正）
    change_points = is_negative_series.ne(is_negative_series.shift(1)).fillna(True)
    
    # 创建一个组标识
    group_id = 0
    group_ids = []
    current_group = 0
    
    # 为每个区间分配唯一ID
    for is_negative, is_change in zip(is_negative_series, change_points):
        if is_change:
            if is_negative:  # 从正变为负，开始新的负区间
                group_id += 1
                current_group = group_id
            else:  # 从负变为正，MACD为正时区间ID为0
                current_group = 0
        
        group_ids.append(current_group)
    
    # 为数据添加组ID
    stock_zh_a_hist_df_copy = stock_zh_a_hist_df.copy()
    stock_zh_a_hist_df_copy["macd_group"] = group_ids
    
    # 计算每个负MACD组内满足所有条件的K线数量
    negative_groups = stock_zh_a_hist_df_copy[stock_zh_a_hist_df_copy["macd_group"] > 0]
    group_counts = {}
    groups_with_data = {}
    
    if not negative_groups.empty:
        # 对每个MACD为负的组进行分析
        for group_id in negative_groups["macd_group"].unique():
            group_data = negative_groups[negative_groups["macd_group"] == group_id]
            # 分别计算两个条件
            # 显示每个区间的详细信息
            print("\n每个MACD为负的区间详情:")
            for group_id, stats in group_counts.items():
                print(f"区间ID: {group_id}, 总K线数: {stats['total_days']}, 满足条件K线数: {stats['condition_days']}")
            
            # 计算各种可能的分布情况
            condition_counts = [g["condition_days"] for g in group_counts.values() if g["condition_days"] > 0]
            counter = Counter(condition_counts)
            
            print("\n满足条件K线分布情况:")
            for count, freq in sorted(counter.items()):
                print(f"含有{count}根满足条件K线的区间有: {freq}个")
            
            # 计算满足条件的比例
            ratios = []
            for g in group_counts.values():
                if g["condition_days"] > 0:
                    ratio = g["condition_days"] / g["total_days"] * 100
                    ratios.append(ratio)
            
            if ratios:
                avg_ratio = sum(ratios) / len(ratios)
                print(f"\n在包含满足条件K线的区间中，满足条件K线平均占区间总K线数的: {avg_ratio:.2f}%")
            
            # 绘制满足条件K线数量的分布图
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            counts_list = [g["condition_days"] for g in group_counts.values()]
            plt.hist(counts_list, bins=range(0, max(counts_list)+2), alpha=0.7, color="blue")
            plt.title("MACD为负区间中满足条件K线数量分布")
            plt.xlabel("满足条件的K线数量")
            plt.ylabel("区间个数")
            plt.grid(True, alpha=0.3)
            
            # 绘制区间长度与满足条件K线数量的散点图
            plt.subplot(1, 2, 2)
            x = [g["total_days"] for g in group_counts.values()]
            y = [g["condition_days"] for g in group_counts.values()]
            plt.scatter(x, y, alpha=0.6, color="red")
            plt.title("区间长度 vs 满足条件K线数量")
            plt.xlabel("区间总K线数")
            plt.ylabel("满足条件的K线数量")
            plt.grid(True, alpha=0.3)
            
            # 添加线性回归线
            if len(x) > 1:  # 至少需要两个点才能绘制回归线
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                plt.plot(x, p(x), "r--", alpha=0.8)
                corr = np.corrcoef(x, y)[0, 1]
                plt.text(0.05, 0.95, f"相关系数: {corr:.2f}", transform=plt.gca().transAxes)
            
            plt.tight_layout()
            plt.savefig("macd_negative_periods_analysis.png")
            print("\n分析图表已保存为 'macd_negative_periods_analysis.png'")
        else:
            print("没有MACD为负的区间包含满足所有条件的K线")
    else:
        print("没有找到MACD连续为负的区间")

# 执行分析
analyze_consecutive_negative_macd()
