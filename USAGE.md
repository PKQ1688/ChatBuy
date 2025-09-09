# ChatBuy 使用指南

## 快速开始

### 1. 运行交互式界面

```bash
# 激活虚拟环境
source .venv/bin/activate

# 运行交互式界面
uv run python scripts/chatbuy_demo.py
```

### 2. 运行组件测试

```bash
# 测试所有组件
uv run python scripts/test_components.py
```

## 使用方法

### 支持的策略描述

系统目前支持双均线策略的自然语言描述：

```
双均线金叉买入，20日均线和50日均线
快线10日，慢线30日，金叉买入死叉卖出
短期均线交叉长期均线，快线20慢线60
```

### 交互命令

- `help` - 显示帮助信息
- `examples` - 显示示例策略描述
- `quit/exit/q` - 退出程序

## 系统架构

```
chatbuy/
├── nlp/                    # 自然语言处理模块
│   ├── strategy_parser.py  # 策略解析器
│   ├── intent_classifier.py # 意图分类
│   └── entity_extractor.py  # 实体提取
├── strategies/             # 策略生成模块
│   ├── strategy_factory.py # 策略工厂
│   ├── base_strategy.py    # 基础策略类
│   └── templates/          # 策略模板
│       └── moving_average.py # 双均线策略
├── backtest/              # 回测执行模块
│   ├── engine.py          # 回测引擎
│   └── vectorbt_wrapper.py # vectorbt封装
├── data/                  # 数据管理模块
│   ├── fetcher.py         # 数据获取
│   └── processor.py       # 数据预处理
└── ui/                    # 用户交互模块
    └── cli.py             # 命令行界面
```

## 功能特点

1. **自然语言处理**: 将中文策略描述转换为可执行的策略参数
2. **策略生成**: 基于参数自动生成交易策略
3. **数据获取**: 支持从Yahoo Finance获取实时数据
4. **回测引擎**: 使用vectorbt进行高性能回测
5. **可视化**: 提供丰富的回测结果展示

## 扩展开发

### 添加新策略

1. 在 `chatbuy/strategies/templates/` 下创建新的策略文件
2. 继承 `BaseStrategy` 类
3. 实现 `generate_signals` 和 `calculate_indicators` 方法
4. 在 `StrategyFactory` 中注册新策略

### 添加新的NLP模式

1. 在 `intent_classifier.py` 中添加新的策略模式
2. 在 `entity_extractor.py` 中添加参数提取规则
3. 在 `strategy_parser.py` 中处理新的策略类型

## 依赖项

- vectorbt: 回测引擎
- yfinance: 数据获取
- pandas: 数据处理
- rich: 美化CLI输出
- ccxt: 加密货币交易（可选）

## 注意事项

- 目前仅支持双均线策略，后续会扩展更多策略类型
- 数据获取需要网络连接
- 回测结果仅供参考，不构成投资建议