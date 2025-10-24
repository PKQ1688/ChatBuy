# ChatBuy 使用指南

## 快速开始

### 1. 运行界面

#### 方式 A: Web 界面（推荐）

```bash
# 激活虚拟环境
source .venv/bin/activate

# 运行 Gradio Web 界面
uv run python scripts/run_gradio_app.py

# 浏览器会自动打开 http://localhost:7860
```

**Web 界面优势：**
- 🎨 可视化图表展示
- 📊 实时策略预览
- 💡 一键示例填充
- 📈 交互式权益曲线
- 🔍 详细交易记录查看

#### 方式 B: 命令行界面

```bash
# 运行 CLI 交互式界面
uv run python scripts/chatbuy_demo.py
```

### 2. 运行测试

```bash
# 测试所有组件
uv run python scripts/test_components.py

# 测试 Gradio 界面
uv run python scripts/test_gradio.py
```

## 使用方法

### Web 界面使用流程

1. **输入策略描述**：在左侧文本框输入中文策略描述
2. **选择数据配置**：选择交易对（如 BTC-USD）和时间范围
3. **查看实时预览**：右侧会显示策略解析结果和置信度
4. **点击开始回测**：系统自动执行回测并展示结果
5. **分析回测结果**：查看性能指标、权益曲线和交易记录

详细使用说明请参考：[Gradio UI 使用指南](GRADIO_GUIDE.md)

### 支持的策略描述

系统支持多种策略的自然语言描述：

**双均线策略：**
```
双均线金叉买入，20日均线和50日均线
快线10日，慢线30日，金叉买入死叉卖出
5日均线上穿20日均线时买入，下穿时卖出
```

**RSI 策略：**
```
RSI低于30买入，高于70卖出
RSI超卖买入，超买卖出
```

**MACD 策略：**
```
MACD金叉买入，死叉卖出
MACD线上穿信号线买入
```

### CLI 交互命令

命令行界面支持以下命令：

- `help` - 显示帮助信息
- `examples` - 显示示例策略描述
- `quit/exit/q` - 退出程序

## 系统架构

```
chatbuy/
├── nlp/                    # 自然语言处理模块
│   ├── strategy_parser.py    # 策略解析器（使用 LLM）
│   ├── intent_classifier.py  # 意图分类
│   └── entity_extractor.py   # 实体提取
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
    ├── cli.py             # 命令行界面
    ├── gradio_app.py      # Web 界面（Gradio）
    ├── components/        # UI 组件
    └── utils/             # UI 工具函数
```

## 功能特点

1. **自然语言处理**: 使用 AI (OpenAI API) 将中文策略描述转换为可执行参数
2. **动态策略生成**: 支持任意买卖条件的策略自动生成
3. **双界面支持**: Web 界面（Gradio）+ 命令行界面（CLI）
4. **数据获取**: 支持从 Yahoo Finance 获取历史数据
5. **高性能回测**: 使用 vectorbt 进行向量化回测
6. **交互式可视化**: Plotly 图表、实时预览、性能仪表板
7. **实时策略预览**: 输入即时显示解析结果和置信度

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

## 性能指标说明

### 关键指标解读

- **总收益率**: 整个回测期间的累计收益百分比
- **年化收益率**: 按年计算的平均收益率
- **夏普比率**: 风险调整后的收益（>1 良好，>2 优秀）
- **最大回撤**: 从峰值到谷底的最大跌幅
- **胜率**: 盈利交易占总交易的比例

## 注意事项

- 需要配置 OpenAI API 密钥（在 `.env` 文件中设置）
- 数据获取需要网络连接
- 回测结果仅供学习参考，不构成投资建议
- 历史表现不代表未来收益
- 实盘前请充分测试和风险评估

## 相关文档

- [Gradio Web UI 详细指南](GRADIO_GUIDE.md) - Web 界面完整使用说明
- [开发者指南](CLAUDE.md) - 开发和扩展指南
- [项目说明](README.md) - 项目概述和快速开始
