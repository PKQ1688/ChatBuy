import pandas as pd

from ..base_strategy import BaseStrategy, StrategyParamValue


class DynamicStrategy(BaseStrategy):
    """动态策略生成器，支持任意买卖条件."""

    def __init__(self, parameters: dict[str, StrategyParamValue]):
        super().__init__(parameters)
        self.buy_conditions = parameters.get("buy_conditions", [])
        self.sell_conditions = parameters.get("sell_conditions", [])
        self.indicators_needed = parameters.get("indicators_needed", {})
        self.strategy_description = parameters.get("strategy_description", "")

    def validate_parameters(self) -> bool:
        """验证参数."""
        return (
            isinstance(self.buy_conditions, list)
            and isinstance(self.sell_conditions, list)
            and isinstance(self.indicators_needed, dict)
            and len(self.buy_conditions) > 0
            and len(self.sell_conditions) > 0
        )

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算所需的技术指标."""
        df = data.copy()

        # 计算移动平均线
        for period in self.indicators_needed.get("ma_periods", []):
            df[f"ma_{period}"] = df["Close"].rolling(window=period).mean()

        # 计算RSI
        for period in self.indicators_needed.get("rsi_periods", []):
            rsi_series: pd.Series = df["Close"]  # type: ignore[assignment]
            df[f"rsi_{period}"] = self._calculate_rsi(rsi_series, period)

        # 计算布林带
        for period in self.indicators_needed.get("bb_periods", []):
            close_series: pd.Series = df["Close"]  # type: ignore[assignment]
            bb_upper, bb_lower = self._calculate_bollinger_bands(close_series, period)
            df[f"bb_upper_{period}"] = bb_upper
            df[f"bb_lower_{period}"] = bb_lower
            df[f"bb_middle_{period}"] = df["Close"].rolling(window=period).mean()

        # 计算MACD
        if self.indicators_needed.get("macd", False):
            close_series: pd.Series = df["Close"]  # type: ignore[assignment]
            macd_line, signal_line, histogram = self._calculate_macd(close_series)
            df["macd_line"] = macd_line
            df["macd_signal"] = signal_line
            df["macd_histogram"] = histogram

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """根据买卖条件生成交易信号."""
        df = self.calculate_indicators(data)
        signals = pd.Series(0, index=df.index)

        # 生成买入信号
        for condition in self.buy_conditions:
            buy_mask = self._evaluate_condition(df, condition)
            signals[buy_mask] = 1

        # 生成卖出信号
        for condition in self.sell_conditions:
            sell_mask = self._evaluate_condition(df, condition)
            signals[sell_mask] = -1

        return signals

    def _evaluate_condition(self, df: pd.DataFrame, condition: dict) -> pd.Series:
        """评估单个条件."""
        indicator = condition["indicator"]
        operator = condition["operator"]

        if indicator not in df.columns:
            return pd.Series(False, index=df.index)

        indicator_values = df[indicator]

        if operator == ">":
            value = condition["value"]
            return indicator_values > value
        elif operator == "<":
            value = condition["value"]
            return indicator_values < value
        elif operator == ">=":
            value = condition["value"]
            return indicator_values >= value
        elif operator == "<=":
            value = condition["value"]
            return indicator_values <= value
        elif operator == "==":
            value = condition["value"]
            return indicator_values == value
        elif operator == "cross_above":
            # 判断是否向上穿越
            value = condition["value"]
            return (indicator_values > value) & (indicator_values.shift(1) <= value)
        elif operator == "cross_below":
            # 判断是否向下穿越
            value = condition["value"]
            return (indicator_values < value) & (indicator_values.shift(1) >= value)
        elif operator == "crosses_above":
            # 判断两条线交叉
            other_indicator = condition.get("other_indicator")
            if other_indicator and other_indicator in df.columns:
                return (indicator_values > df[other_indicator]) & (
                    indicator_values.shift(1) <= df[other_indicator].shift(1)
                )
            return pd.Series(False, index=df.index)
        elif operator == "crosses_below":
            # 判断两条线交叉
            other_indicator = condition.get("other_indicator")
            if other_indicator and other_indicator in df.columns:
                return (indicator_values < df[other_indicator]) & (
                    indicator_values.shift(1) >= df[other_indicator].shift(1)
                )
            return pd.Series(False, index=df.index)

        return pd.Series(False, index=df.index)

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi_result = 100 - (100 / (1 + rs))
        return pd.Series(rsi_result)

    def _calculate_bollinger_bands(
        self, prices: pd.Series, period: int = 20, std_dev: float = 2
    ) -> tuple:
        """计算布林带."""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, lower

    def _calculate_macd(
        self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> tuple:
        """计算MACD."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def get_info(
        self,
    ) -> dict[
        str,
        str
        | list[dict[str, str | int | float]]
        | dict[str, list[int]]
        | dict[str, StrategyParamValue],
    ]:
        """获取策略信息."""
        info = super().get_info()
        info.update(
            {
                "strategy_description": self.strategy_description,
                "buy_conditions": self.buy_conditions,
                "sell_conditions": self.sell_conditions,
                "indicators_needed": self.indicators_needed,
            }
        )
        return info


class StrategyConditionParser:
    """策略条件解析器，将自然语言条件转换为可执行的条件."""

    @staticmethod
    def parse_buy_sell_conditions(
        text: str,
    ) -> dict[str, str | list[dict[str, str | int | float]] | dict[str, list[int]]]:
        """解析买卖条件文本."""
        # 这里使用LLM来解析自然语言条件
        # 为了演示，我们先提供一个简化的解析逻辑

        # 示例：解析"5日均线上穿20日均线时买入，下穿时卖出"
        if "上穿" in text and "下穿" in text:
            # 提取均线周期
            import re

            numbers = re.findall(r"\d+", text)
            if len(numbers) >= 2:
                fast_period = int(numbers[0])
                slow_period = int(numbers[1])

                return {
                    "buy_conditions": [
                        {
                            "indicator": f"ma_{fast_period}",
                            "operator": "crosses_above",
                            "other_indicator": f"ma_{slow_period}",
                        }
                    ],
                    "sell_conditions": [
                        {
                            "indicator": f"ma_{fast_period}",
                            "operator": "crosses_below",
                            "other_indicator": f"ma_{slow_period}",
                        }
                    ],
                    "indicators_needed": {"ma_periods": [fast_period, slow_period]},
                    "strategy_description": text,
                }

        # 示例：解析"RSI低于30买入，高于70卖出"
        if "RSI" in text and "买入" in text and "卖出" in text:
            import re

            numbers = re.findall(r"\d+", text)
            rsi_lower = 30
            rsi_upper = 70
            rsi_period = 14

            if len(numbers) >= 2:
                rsi_lower = int(numbers[0])
                rsi_upper = int(numbers[1])
            elif len(numbers) == 1:
                if "低于" in text:
                    rsi_lower = int(numbers[0])
                else:
                    rsi_upper = int(numbers[0])

            return {
                "buy_conditions": [
                    {
                        "indicator": f"rsi_{rsi_period}",
                        "operator": "<",
                        "value": rsi_lower,
                    }
                ],
                "sell_conditions": [
                    {
                        "indicator": f"rsi_{rsi_period}",
                        "operator": ">",
                        "value": rsi_upper,
                    }
                ],
                "indicators_needed": {"rsi_periods": [rsi_period]},
                "strategy_description": text,
            }

        # 默认返回双均线策略
        return {
            "buy_conditions": [
                {
                    "indicator": "ma_20",
                    "operator": "crosses_above",
                    "other_indicator": "ma_50",
                }
            ],
            "sell_conditions": [
                {
                    "indicator": "ma_20",
                    "operator": "crosses_below",
                    "other_indicator": "ma_50",
                }
            ],
            "indicators_needed": {"ma_periods": [20, 50]},
            "strategy_description": text,
        }
