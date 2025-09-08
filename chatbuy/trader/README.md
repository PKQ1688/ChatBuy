# BTC和ETH轮动策略

这是一个基于技术指标的BTC和ETH轮动交易策略实现。策略通过分析多个技术指标来决定在BTC和ETH之间进行轮动，以获取更好的风险调整收益。

## 策略概述

### 核心思想
- **轮动策略**: 在BTC和ETH之间动态切换，持有表现更好的资产
- **多指标融合**: 结合动量、RSI、MACD、相对强度等多个技术指标
- **风险管理**: 通过分散化和动态调整降低单一资产风险

### 技术指标
1. **动量指标**: 比较BTC和ETH的价格动量
2. **RSI指标**: 选择相对超卖的资产
3. **MACD指标**: 基于趋势跟随信号
4. **相对强度**: 比较两个资产的相对表现
5. **风险调整收益**: 考虑波动率的收益比较
6. **综合信号**: 通过投票机制综合所有信号

## 文件结构

```
trader/
├── __init__.py              # 模块初始化
├── data_manager.py          # 数据获取和管理
├── portfolio.py             # 投资组合管理
├── rotation_strategy.py     # 轮动策略核心实现
├── example_rotation.py      # 使用示例
└── README.md               # 说明文档
```

## 快速开始

### 1. 安装依赖

```bash
pip install pandas numpy ccxt matplotlib
```

### 2. 运行回测

```bash
cd chatbuy/trader
python example_rotation.py --mode backtest
```

### 3. 分析信号

```bash
python example_rotation.py --mode analyze
```

## 详细使用

### 基本用法

```python
from chatbuy.trader import BTCETHRotationStrategy

# 创建策略实例
strategy = BTCETHRotationStrategy(
    initial_balance=10000.0,      # 初始资金
    lookback_period=20,           # 技术指标回望期
    momentum_period=10,           # 动量计算期
    rsi_period=14,               # RSI计算期
    rebalance_frequency="weekly"  # 再平衡频率
)

# 运行回测
results = strategy.backtest(days=365)

# 查看结果
print(f"总收益率: {results['strategy_performance']['total_return_pct']:.2f}%")
```

### 参数说明

#### BTCETHRotationStrategy参数

- `initial_balance`: 初始资金（默认10000 USDT）
- `lookback_period`: 技术指标计算的回望期（默认20天）
- `momentum_period`: 动量指标计算期（默认10天）
- `rsi_period`: RSI指标计算期（默认14天）
- `rebalance_frequency`: 再平衡频率
  - `"daily"`: 每日检查
  - `"weekly"`: 每周检查（默认）
  - `"monthly"`: 每月检查

#### backtest方法参数

- `start_date`: 回测开始日期（ISO格式）
- `end_date`: 回测结束日期（ISO格式）
- `days`: 回测天数（如果未指定具体日期，默认365天）

## 策略逻辑

### 信号生成

1. **动量信号**: 比较BTC和ETH的N日动量，选择动量更强的
2. **RSI信号**: 选择RSI值更低（更超卖）的资产
3. **MACD信号**: 选择MACD信号更强的资产
4. **相对强度信号**: 选择相对表现更好的资产
5. **风险调整信号**: 选择风险调整后收益更高的资产

### 综合决策

使用投票机制综合所有信号：
- 每个信号投票选择BTC或ETH
- 得票多的资产成为目标持仓
- 当目标资产改变或到达再平衡时间时执行交易

### 交易执行

1. 卖出当前所有持仓
2. 用95%的资金买入目标资产（保留5%现金缓冲）
3. 记录交易和投资组合状态

## 结果分析

### 性能指标

- **总收益率**: 策略的累计收益率
- **基准收益率**: 50% BTC + 50% ETH买入持有的收益率
- **超额收益**: 策略收益率 - 基准收益率
- **最大回撤**: 投资组合价值的最大跌幅
- **夏普比率**: 风险调整后的收益指标
- **交易次数**: 总的买卖交易次数

### 输出文件

运行回测后会生成以下文件：

- `data/rotation_portfolio_history.csv`: 投资组合历史记录
- `data/rotation_trade_history.csv`: 交易历史记录
- `data/rotation_signals.csv`: 信号历史记录
- `data/rotation_strategy_results.png`: 结果可视化图表

## 策略优化建议

### 参数调优

1. **调整再平衡频率**: 
   - 更频繁的再平衡可能捕捉更多机会，但增加交易成本
   - 较低频率的再平衡减少交易成本，但可能错过机会

2. **优化技术指标参数**:
   - 调整RSI、MACD、动量指标的计算期
   - 测试不同的回望期长度

3. **改进信号权重**:
   - 为不同信号分配不同权重
   - 基于历史表现动态调整权重

### 风险管理

1. **止损机制**: 添加最大回撤限制
2. **仓位管理**: 实现动态仓位调整
3. **市场状态识别**: 在不同市场环境下使用不同策略

## 注意事项

1. **数据质量**: 确保有稳定的数据源
2. **交易成本**: 实际交易中需要考虑手续费和滑点
3. **市场风险**: 加密货币市场波动性较大
4. **回测偏差**: 历史表现不代表未来收益

## 扩展功能

### 添加更多资产

```python
# 可以扩展为多资产轮动
symbols = ["BTC", "ETH", "BNB", "ADA"]
```

### 集成实盘交易

```python
# 可以集成交易所API进行实盘交易
# 需要添加订单管理和风险控制
```

### 机器学习优化

```python
# 可以使用机器学习方法优化信号权重
# 或预测最佳轮动时机
```

## 免责声明

本策略仅供学习和研究使用，不构成投资建议。加密货币投资存在高风险，可能导致本金损失。在进行实际投资前，请充分了解相关风险并咨询专业投资顾问。