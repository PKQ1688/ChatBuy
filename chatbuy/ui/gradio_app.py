"""Gradio web interface for ChatBuy."""

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from ..backtest.engine import BacktestEngine
from ..data.fetcher import DataFetcher
from ..data.processor import DataProcessor
from ..nlp.strategy_parser import StrategyParser
from ..strategies.strategy_factory import StrategyFactory


class ChatBuyGradio:
    """Gradio web application for ChatBuy."""

    def __init__(self):
        """Initialize the Gradio app with all backend components."""
        self.strategy_parser = StrategyParser()
        self.strategy_factory = StrategyFactory()
        self.backtest_engine = BacktestEngine()
        self.data_fetcher = DataFetcher()
        self.data_processor = DataProcessor()

    def preview_strategy(self, strategy_desc: str) -> dict:
        """Preview strategy parsing results in real-time."""
        if not strategy_desc or len(strategy_desc.strip()) == 0:
            return {"提示": "请输入策略描述"}

        try:
            config = self.strategy_parser.parse(strategy_desc)
            if config:
                return {
                    "策略类型": config["strategy_type"],
                    "参数": str(config["parameters"]),
                    "置信度": f"{config['confidence']:.1%}",
                }
            return {"错误": "无法解析策略描述，请尝试其他描述方式"}
        except Exception as e:
            return {"错误": str(e)}

    def run_backtest(
        self,
        strategy_desc: str,
        symbol: str,
        start_date: str,
        end_date: str,
        initial_cash: float,
        fees: float,
        slippage: float,
    ) -> tuple:
        """Execute backtest and return formatted results."""
        try:
            # 1. Parse strategy
            config = self.strategy_parser.parse(strategy_desc)
            if not config:
                return (
                    {"错误": "无法解析策略描述"},
                    None,
                    None,
                    None,
                    "❌ 策略解析失败",
                )

            # 2. Create strategy
            strategy = self.strategy_factory.create_strategy(
                config["strategy_type"], config["parameters"]
            )
            if not strategy:
                return (
                    {"错误": "无法创建策略实例"},
                    None,
                    None,
                    None,
                    "❌ 策略创建失败",
                )

            # 3. Fetch data
            data = self.data_fetcher.fetch_yfinance(
                symbol, start_date=start_date, end_date=end_date
            )
            if data is None or len(data) == 0:
                return (
                    {"错误": "无法获取数据"},
                    None,
                    None,
                    None,
                    "❌ 数据获取失败",
                )

            # 4. Clean data
            data = self.data_processor.clean_data(data)
            if not self.data_processor.validate_data(data):
                return (
                    {"错误": "数据验证失败"},
                    None,
                    None,
                    None,
                    "❌ 数据验证失败",
                )

            # 5. Run backtest
            results = self.backtest_engine.run_strategy_backtest(
                strategy, data, initial_cash, fees, slippage
            )
            if not results:
                return (
                    {"错误": "回测执行失败"},
                    None,
                    None,
                    None,
                    "❌ 回测执行失败",
                )

            # 6. Format outputs
            strategy_info = self._format_strategy_info(config, results)
            metrics_df = self._format_metrics(results)
            equity_plot = self._create_equity_plot(results)
            trades_df = self._format_trades(results)

            status_msg = (
                f"✅ 回测完成！交易对: {symbol} | 周期: {start_date} ~ {end_date}"
            )

            return strategy_info, metrics_df, equity_plot, trades_df, status_msg

        except Exception as e:
            return (
                {"错误": str(e)},
                None,
                None,
                None,
                f"❌ 错误: {str(e)}",
            )

    def _format_strategy_info(self, config: dict, results: dict) -> dict:
        """Format strategy information for display."""
        strategy = results.get("strategy", {})
        meta = results.get("meta", {})

        info = {
            "策略名称": strategy.get("name", "未知"),
            "策略类型": config.get("strategy_type", "未知"),
            "策略描述": strategy.get("description", "无"),
            "参数": str(config.get("parameters", {})),
            "置信度": f"{config.get('confidence', 0):.1%}",
            "交易对": meta.get("symbol", "未知"),
        }

        return info

    def _format_metrics(self, results: dict) -> pd.DataFrame:
        """Format performance metrics as DataFrame."""
        stats = results.get("stats", {})

        # Create metrics table
        data = {
            "指标": [
                "总收益率",
                "年化收益率",
                "夏普比率",
                "最大回撤",
                "胜率",
                "总交易次数",
                "盈利交易",
                "亏损交易",
            ],
            "数值": [
                f"{stats.get('total_return', 0):.2%}",
                f"{stats.get('annualized_return', 0):.2%}",
                f"{stats.get('sharpe_ratio', 0):.2f}",
                f"{stats.get('max_drawdown', 0):.2%}",
                f"{stats.get('win_rate', 0):.2%}",
                str(stats.get("total_trades", 0)),
                str(int(stats.get("total_trades", 0) * stats.get("win_rate", 0))),
                str(int(stats.get("total_trades", 0) * (1 - stats.get("win_rate", 0)))),
            ],
        }

        return pd.DataFrame(data)

    def _create_equity_plot(self, results: dict) -> go.Figure:
        """Create equity curve plot using Plotly."""
        portfolio = results.get("portfolio")
        if portfolio is None:
            return go.Figure()

        # Get equity data with error handling
        try:
            equity = portfolio.value()
            if equity is None or len(equity) == 0:
                return go.Figure()
        except Exception:
            return go.Figure()

        # Create figure
        fig = go.Figure()

        # Add equity curve
        try:
            fig.add_trace(
                go.Scatter(
                    x=equity.index,
                    y=equity.values,
                    mode="lines",
                    name="权益曲线",
                    line=dict(color="#2E86AB", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(46, 134, 171, 0.1)",
                )
            )
        except Exception:
            return go.Figure()

        # Update layout
        fig.update_layout(
            title="账户权益曲线",
            xaxis_title="日期",
            yaxis_title="账户价值",
            template="plotly_white",
            hovermode="x unified",
            height=400,
        )

        return fig

    def _format_trades(self, results: dict) -> pd.DataFrame:
        """Format trade records as DataFrame."""
        trades = results.get("trades")
        if trades is None:
            return pd.DataFrame({"提示": ["暂无交易记录"]})

        # Check if trades has records_readable attribute (vectorbt format)
        if hasattr(trades, 'records_readable') and len(trades.records_readable) > 0:
            df = trades.records_readable.copy()

            # Rename columns to Chinese
            columns_map = {
                "Entry Timestamp": "进入时间",
                "Exit Timestamp": "退出时间",
                "Avg Entry Price": "买入价格",
                "Avg Exit Price": "卖出价格",
                "PnL": "盈亏",
                "Return": "收益率",
                "Size": "数量",
                "Direction": "方向",
                "Status": "状态"
            }

            # Select and rename available columns
            available_cols = [col for col in columns_map.keys() if col in df.columns]
            if available_cols:
                df = df[available_cols].rename(columns=columns_map)

            # Format timestamps
            if "进入时间" in df.columns:
                df["进入时间"] = pd.to_datetime(df["进入时间"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            if "退出时间" in df.columns:
                df["退出时间"] = pd.to_datetime(df["退出时间"]).dt.strftime("%Y-%m-%d %H:%M:%S")

            # Format numeric columns
            for col in ["买入价格", "卖出价格", "盈亏", "收益率", "数量"]:
                if col in df.columns:
                    df[col] = df[col].round(4)

            return df.head(20)  # Show only first 20 trades
        else:
            return pd.DataFrame({"提示": ["暂无交易记录"]})

    def build_app(self) -> gr.Blocks:
        """Build the Gradio interface."""
        with gr.Blocks(
            theme=gr.themes.Soft(),
            title="ChatBuy - AI 量化交易系统",
            css="""
            .header {text-align: center; padding: 20px;}
            .status-box {padding: 10px; border-radius: 5px; margin: 10px 0;}
            """,
        ) as app:
            # Header
            gr.Markdown(
                """
                # 🤖 ChatBuy - AI 量化交易系统
                ### 用自然语言描述策略，AI 自动生成回测
                """
            )

            with gr.Tabs():
                # Tab 1: Strategy Creation and Backtest
                with gr.Tab("📊 策略回测"):
                    with gr.Row():
                        # Left column: Input area
                        with gr.Column(scale=1):
                            gr.Markdown("### 📝 策略描述")

                            strategy_input = gr.Textbox(
                                label="输入您的交易策略",
                                placeholder="例如：双均线金叉买入，5日和20日均线\n或：RSI低于30买入，高于70卖出",
                                lines=5,
                            )

                            # Quick examples
                            gr.Examples(
                                examples=[
                                    ["双均线金叉买入，死叉卖出。5日均线和20日均线"],
                                    ["RSI低于30买入，高于70卖出"],
                                    ["MACD金叉买入，死叉卖出"],
                                    ["10日均线上穿30日均线时买入，下穿时卖出"],
                                ],
                                inputs=strategy_input,
                                label="💡 示例策略（点击使用）",
                            )

                            gr.Markdown("### 📈 数据配置")

                            symbol = gr.Dropdown(
                                choices=[
                                    "BTC-USD",
                                    "ETH-USD",
                                    "AAPL",
                                    "TSLA",
                                    "NVDA",
                                    "MSFT",
                                ],
                                value="BTC-USD",
                                label="交易对",
                            )

                            with gr.Row():
                                start_date = gr.Textbox(
                                    value="2024-01-01", label="开始日期 (YYYY-MM-DD)"
                                )
                                end_date = gr.Textbox(
                                    value="2024-06-01", label="结束日期 (YYYY-MM-DD)"
                                )

                            with gr.Accordion("⚙️ 高级参数", open=False):
                                initial_cash = gr.Number(
                                    value=10000, label="初始资金 ($)"
                                )
                                fees = gr.Slider(
                                    0, 0.01, value=0.001, label="手续费率", step=0.0001
                                )
                                slippage = gr.Slider(
                                    0, 0.01, value=0.001, label="滑点", step=0.0001
                                )

                            run_btn = gr.Button(
                                "🚀 开始回测", variant="primary", size="lg"
                            )

                        # Right column: Preview and results
                        with gr.Column(scale=2):
                            gr.Markdown("### 📊 策略预览")
                            strategy_info = gr.JSON(label="策略信息")

                            status_box = gr.Textbox(
                                label="状态", value="等待输入...", interactive=False
                            )

                    # Results section
                    gr.Markdown("---")
                    gr.Markdown("## 📈 回测结果")

                    with gr.Row():
                        metrics_table = gr.Dataframe(
                            label="性能指标", interactive=False
                        )

                    equity_plot = gr.Plot(label="权益曲线")

                    trades_table = gr.Dataframe(
                        label="交易记录（最近20条）", interactive=False
                    )

                    # Event handlers
                    strategy_input.change(
                        fn=self.preview_strategy,
                        inputs=strategy_input,
                        outputs=strategy_info,
                    )

                    run_btn.click(
                        fn=self.run_backtest,
                        inputs=[
                            strategy_input,
                            symbol,
                            start_date,
                            end_date,
                            initial_cash,
                            fees,
                            slippage,
                        ],
                        outputs=[
                            strategy_info,
                            metrics_table,
                            equity_plot,
                            trades_table,
                            status_box,
                        ],
                    )

                # Tab 2: Help
                with gr.Tab("📚 帮助文档"):
                    gr.Markdown(
                        """
                    # 使用指南

                    ## 🚀 快速开始

                    1. **描述策略**：在左侧文本框中用中文描述您的交易策略
                    2. **选择数据**：选择交易对和回测时间范围
                    3. **运行回测**：点击"开始回测"按钮
                    4. **查看结果**：分析性能指标、权益曲线和交易记录

                    ## 📝 支持的策略类型

                    ### 1. 双均线策略
                    - `"双均线金叉买入，5日和20日均线"`
                    - `"快线10日，慢线30日，金叉买入死叉卖出"`
                    - `"10日均线上穿30日均线时买入"`

                    ### 2. RSI 策略
                    - `"RSI低于30买入，高于70卖出"`
                    - `"RSI超卖买入，超买卖出"`

                    ### 3. MACD 策略
                    - `"MACD金叉买入，死叉卖出"`
                    - `"MACD线上穿信号线买入"`

                    ## 📊 性能指标说明

                    - **总收益率**：整个回测期间的累计收益
                    - **年化收益率**：按年计算的平均收益率
                    - **夏普比率**：风险调整后的收益，越高越好（>1为良好）
                    - **最大回撤**：从峰值到谷底的最大跌幅
                    - **胜率**：盈利交易占总交易的比例

                    ## ⚠️ 注意事项

                    - 回测结果仅供参考，不构成投资建议
                    - 历史表现不代表未来收益
                    - 请在实盘前充分测试和验证策略
                    - 注意控制风险，合理配置资金

                    ## 🔧 技术支持

                    - GitHub: https://github.com/PKQ1688/ChatBuy
                    - 问题反馈：提交 Issue
                    """
                    )

            return app

    def launch(self, **kwargs):
        """Launch the Gradio app."""
        app = self.build_app()
        app.launch(**kwargs)
