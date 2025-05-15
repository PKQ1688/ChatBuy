import numpy as np
import pandas as pd
import vectorbt as vbt


class BaseStrategy:
    """Base class for backtesting strategies."""

    vbt.settings.portfolio["init_cash"] = 10000.0  # 10000$
    vbt.settings.portfolio["fees"] = 0.001  # 0.1%
    vbt.settings.portfolio["slippage"] = 0.0025  # 0.25%

    def __init__(
        self, symbol="BTC-USD", start="2022-01-01 UTC", end="2024-01-01 UTC"
    ):  # 更新默认日期范围
        self.symbol = symbol
        self.start = start
        self.end = end
        data = vbt.YFData.download(self.symbol, start=self.start, end=self.end)
        self.price = data.get("Close")
        self.high = data.get("High")
        self.low = data.get("Low")
        self.open = data.get("Open")
        if self.price is None or self.price.empty:
            print(
                f"警告: 未能加载 {self.symbol} 从 {self.start} 到 {self.end} 的价格数据。"
            )
        self.entries = None
        self.exits = None

    def init_entries_exits(self):
        """Initialize entries and exits for the strategy. This method should be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement init_entries_exits")

    def run(self):
        """Run the backtest and return the portfolio."""
        if self.entries is None or self.exits is None:
            self.init_entries_exits()

        # 确保 entries 和 exits 是布尔类型的 NumPy 数组
        entries_np = (
            self.entries.values.astype(bool)
            if isinstance(self.entries, pd.Series)
            else self.entries.astype(bool)
        )
        exits_np = (
            self.exits.values.astype(bool)
            if isinstance(self.exits, pd.Series)
            else self.exits.astype(bool)
        )

        pf = vbt.Portfolio.from_signals(self.price, entries_np, exits_np)
        return pf


class MaCross(BaseStrategy):
    """A strategy that uses two moving averages to generate buy and sell signals."""

    def __init__(
        self,
        symbol="BTC-USD",
        start="2022-01-01 UTC",
        end="2024-01-01 UTC",
        fast_window=5,
        slow_window=10,
    ):  # 更新默认日期范围
        super().__init__(symbol, start, end)
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.fast_ma = vbt.MA.run(self.price, self.fast_window, short_name="fast")
        self.slow_ma = vbt.MA.run(self.price, self.slow_window, short_name="slow")
        self.init_entries_exits()

    def init_entries_exits(self):
        """Initialize entries and exits for the strategy."""
        self.entries = self.fast_ma.ma_crossed_above(self.slow_ma)
        self.exits = self.fast_ma.ma_crossed_below(self.slow_ma)


class TurtleStrategy(BaseStrategy):
    """Turtle Trading Strategy: Donchian Channel breakout entry, ATR stop-loss exit."""

    def __init__(
        self,
        symbol="BTC-USD",
        start="2022-01-01 UTC",
        end="2024-01-01 UTC",
        entry_window=20,
        exit_window=10,
        atr_window=20,
        risk_pct=0.01,
        atr_sl_multiplier=2.0,  # ATR multiplier for stop-loss
    ):
        super().__init__(symbol, start, end)
        self.entry_window = entry_window
        self.exit_window = exit_window
        self.atr_window = atr_window
        self.risk_pct = risk_pct
        self.atr_sl_multiplier = atr_sl_multiplier

        # 唐奇安通道用 rolling/max/min 计算
        if self.price is not None and not self.price.empty:
            self.dc_entry_upper = self.price.rolling(
                self.entry_window, min_periods=1
            ).max()
            self.dc_exit_lower = self.price.rolling(
                self.exit_window, min_periods=1
            ).min()
        else:
            # Handle case where price data is not available early
            self.dc_entry_upper = pd.Series(dtype=float)
            self.dc_exit_lower = pd.Series(dtype=float)
            print(
                "Warning: Price data is None or empty during TurtleStrategy __init__ for Donchian channels."
            )

        # ATR 需要 high/low/close
        if (
            self.high is not None
            and self.low is not None
            and self.price is not None
            and not self.price.empty
        ):
            self.atr = vbt.ATR.run(
                self.high, self.low, self.price, window=self.atr_window
            ).atr
        else:
            self.atr = None
            # print("Warning: Price data might be empty or H/L/C not available for ATR calculation in TurtleStrategy __init__.")

        self.sl_stop_values = pd.Series(dtype=float)
        self.init_entries_exits()

    def init_entries_exits(self):
        if self.price is None or self.price.empty or self.dc_entry_upper.empty:
            idx = (
                pd.RangeIndex(start=0, stop=0, step=1)
                if self.price is None or self.price.empty
                else self.price.index
            )
            self.entries = pd.Series([False] * len(idx), index=idx)
            self.exits = pd.Series([False] * len(idx), index=idx)
            self.sl_stop_values = pd.Series(dtype=float)
            # print("Warning: Price data is None or empty in TurtleStrategy.init_entries_exits. Signals will be empty.")
            return

        # 入场：价格上穿 entry_window 唐奇安通道上轨
        self.entries = self.price > self.dc_entry_upper
        # 出场：价格下穿 exit_window 唐奇安通道下轨 (Donchian-based exit)
        self.exits = self.price < self.dc_exit_lower

        # ATR Stop-Loss calculation
        # self.sl_stop_values should be a Series of stop prices, indexed by the entry datetime.
        current_sl_values = pd.Series(np.nan, index=self.price.index, dtype=float)
        if (
            self.atr is not None
            and not self.atr.empty
            and self.entries is not None
            and not self.entries.empty
        ):
            entry_signal_indices = self.entries[self.entries].index

            if not entry_signal_indices.empty:
                # Ensure indices are present in all relevant series
                valid_entry_indices = entry_signal_indices.intersection(
                    self.price.index
                ).intersection(self.atr.index)

                if not valid_entry_indices.empty:
                    entry_prices_at_signal = self.price.loc[valid_entry_indices]
                    atr_at_signal = self.atr.loc[valid_entry_indices]

                    calculated_sl_prices = entry_prices_at_signal - (
                        self.atr_sl_multiplier * atr_at_signal
                    )
                    current_sl_values.loc[valid_entry_indices] = calculated_sl_prices

                    self.sl_stop_values = (
                        current_sl_values.dropna()
                    )  # Keep only actual stop values at entry points
                    if self.sl_stop_values.empty:
                        self.sl_stop_values = pd.Series(
                            dtype=float
                        )  # Ensure it's an empty series of correct type
                else:
                    self.sl_stop_values = pd.Series(dtype=float)
            else:
                self.sl_stop_values = pd.Series(dtype=float)
        else:
            self.sl_stop_values = pd.Series(dtype=float)
            # if self.atr is None or self.atr.empty:
            #      print("Warning: ATR data is None or empty during TurtleStrategy.init_entries_exits. ATR stop-loss cannot be calculated.")

    def run(self):
        """Run the backtest and return the portfolio."""
        if self.price is None or self.price.empty:
            print(
                f"Error: Price data is empty for {self.symbol} in TurtleStrategy. Cannot run backtest."
            )
            # Attempt to return an empty portfolio object if vectorbt allows
            try:
                return vbt.Portfolio.from_signals(
                    self.price, pd.Series(dtype=bool), pd.Series(dtype=bool)
                )
            except:  # pylint: disable=bare-except
                return None

        if (
            self.entries is None
            or self.exits is None
            or not hasattr(self, "sl_stop_values")
        ):
            self.init_entries_exits()

        common_index = self.price.index

        # Ensure entries and exits are pd.Series aligned with common_index
        if not isinstance(self.entries, pd.Series) or not self.entries.index.equals(
            common_index
        ):
            entries_s = (
                pd.Series(self.entries, index=common_index).reindex(
                    common_index, fill_value=False
                )
                if self.entries is not None
                else pd.Series(False, index=common_index)
            )
        else:
            entries_s = self.entries

        if not isinstance(self.exits, pd.Series) or not self.exits.index.equals(
            common_index
        ):
            exits_s = (
                pd.Series(self.exits, index=common_index).reindex(
                    common_index, fill_value=False
                )
                if self.exits is not None
                else pd.Series(False, index=common_index)
            )
        else:
            exits_s = self.exits

        entries_np = entries_s.values.astype(bool)
        exits_np = exits_s.values.astype(bool)

        sl_param_to_use = None
        if (
            hasattr(self, "sl_stop_values")
            and isinstance(self.sl_stop_values, pd.Series)
            and not self.sl_stop_values.empty
        ):
            # self.sl_stop_values is already a Series indexed by entry timestamps with stop price values.
            # This is the format vectorbt expects for sl_stop when providing per-signal stops.
            sl_param_to_use = self.sl_stop_values

        risk_pct_to_use = None
        if (
            hasattr(self, "risk_pct")
            and self.risk_pct is not None
            and self.risk_pct > 0
        ):
            risk_pct_to_use = self.risk_pct
            if sl_param_to_use is None or sl_param_to_use.empty:
                print(
                    "Warning: risk_pct is set for TurtleStrategy, but ATR stop-loss (sl_stop) is not available or empty. Position sizing via sl_target_percent might not be effective."
                )

        freq = pd.infer_freq(self.price.index)
        if (
            freq is None and len(self.price.index) > 1
        ):  # Try to manually determine if daily
            time_diffs = self.price.index.to_series().diff().dropna()
            if not time_diffs.empty and (time_diffs == pd.Timedelta(days=1)).all():
                freq = "D"
        if freq is None:
            print(
                "Warning: Could not infer frequency for TurtleStrategy. Backtest might be inaccurate. Defaulting to 'D' if data seems daily-like or has some length."
            )
            if len(self.price.index) > 1:  # Only default if there's some data
                freq = "D"

        # print(f"Debug TurtleStrategy run: entries sum: {entries_np.sum()}, exits sum: {exits_np.sum()}")
        # if sl_param_to_use is not None and not sl_param_to_use.empty:
        #     print(f"Debug TurtleStrategy run: sl_param_to_use ({len(sl_param_to_use)}) head: {sl_param_to_use.head()}")
        # else:
        #     print("Debug TurtleStrategy run: sl_param_to_use is None or empty")
        # print(f"Debug TurtleStrategy run: risk_pct_to_use: {risk_pct_to_use}")
        # print(f"Debug TurtleStrategy run: freq: {freq}")
        # print(f"Debug TurtleStrategy run: price data length: {len(self.price)}")
        # print(f"Debug TurtleStrategy run: Portfolio init_cash: {vbt.settings.portfolio['init_cash']}")

        pf = vbt.Portfolio.from_signals(
            self.price,
            entries_np,
            exits_np,
            sl_stop=sl_param_to_use,
            sl_target_percent=risk_pct_to_use,  # This tells vectorbt to size based on stop
            freq=freq,
            # init_cash, fees, slippage are taken from vbt.settings.portfolio
        )
        return pf


class MacdBullishDivergence(BaseStrategy):
    """MACD Divergence Strategy.

    Generates buy signals based on bullish divergence (MACD histogram forms a higher low
    during a death cross phase compared to the previous death cross phase)
    and sell signals based on bearish divergence (MACD histogram forms a lower high
    during a golden cross phase compared to the previous golden cross phase).

    The comparison of MACD histogram phases can be done using different metrics:
    - 'min_max': Compares the minimum histogram value for bullish divergence (higher is weaker/better)
                 and the maximum histogram value for bearish divergence (lower is weaker/better).
    - 'area': Compares the sum of histogram values (area under the curve).
              For bullish, a less negative/more positive sum is weaker/better.
              For bearish, a less positive/more negative sum is weaker/better.
    - 'average': Compares the average of histogram values.
                 Similar interpretation to 'area'.
    """

    def __init__(
        self,
        symbol="BTC-USD",
        start="2022-01-01 UTC",
        end="2024-01-01 UTC",
        comparison_metric: str = "min_max",  # 'min_max', 'area', 'average'
    ):
        super().__init__(symbol, start, end)
        if self.price is None or self.price.empty:
            # BaseStrategy prints a warning. MACD calculation will fail if price is None.
            # Let init_entries_exits handle creating empty signals.
            self.macd_line = None
            self.signal_line = None
            self.macd_hist = None
            self.comparison_metric = comparison_metric  # Still set this
            self.init_entries_exits()  # This will set empty entries/exits
            return

        macd_indicator = vbt.MACD.run(self.price)
        self.macd_line = macd_indicator.macd
        self.signal_line = macd_indicator.signal
        self.macd_hist = macd_indicator.hist
        self.comparison_metric = comparison_metric
        if self.comparison_metric not in ["min_max", "area", "average"]:
            raise ValueError(
                "comparison_metric must be one of 'min_max', 'area', or 'average'"
            )
        self.init_entries_exits()

    def init_entries_exits(self):
        """Initializes entries and exits for the MACD divergence strategy."""
        if self.price is None or self.price.empty or self.macd_hist is None:
            # Create empty Series with the correct index if price is available, else use a default range if price is also None
            idx = (
                self.price.index
                if (self.price is not None and not self.price.empty)
                else pd.RangeIndex(start=0, stop=0, step=1)
            )
            self.entries = pd.Series([False] * len(idx), index=idx)
            self.exits = pd.Series([False] * len(idx), index=idx)
            return

        self.entries = self.detect_macd_divergence(
            buy=True, comparison_metric=self.comparison_metric
        )
        self.exits = self.detect_macd_divergence(
            buy=False, comparison_metric=self.comparison_metric
        )

    def _calculate_metric(self, hist_segment, metric_type, is_buy_signal_check):
        """Helper to calculate metric for a histogram segment."""
        if not isinstance(hist_segment, np.ndarray):
            hist_segment = np.array(hist_segment)
        if len(hist_segment) == 0:
            return np.nan

        if metric_type == "min_max":
            return np.min(hist_segment) if is_buy_signal_check else np.max(hist_segment)
        elif metric_type == "area":
            return np.sum(hist_segment)
        elif metric_type == "average":
            return np.mean(hist_segment)
        return np.nan

    def detect_macd_divergence(self, buy: bool, comparison_metric: str):
        """Detects MACD divergence.

        If buy=True (Bullish Divergence):
            Looks for a death cross phase (death cross to next golden cross).
            A buy signal is generated at the golden cross if the current death cross phase
            is "weaker" (e.g., higher min hist value for 'min_max') than the previous one.
        If buy=False (Bearish Divergence):
            Looks for a golden cross phase (golden cross to next death cross).
            A sell signal is generated at the death cross if the current golden cross phase
            is "weaker" (e.g., lower max hist value for 'min_max') than the previous one.
        """
        macd = self.macd_line.values
        signal = self.signal_line.values
        hist = self.macd_hist.values
        n = len(macd)
        result = pd.Series([False] * n, index=self.price.index)

        death_crosses = []
        golden_crosses = []

        for i in range(1, n):
            if macd[i - 1] > signal[i - 1] and macd[i] <= signal[i]:
                death_crosses.append(i)
            if macd[i - 1] < signal[i - 1] and macd[i] >= signal[i]:
                golden_crosses.append(i)

        if buy:  # Bullish Divergence
            for current_gc_signal_point in golden_crosses:
                current_phase_dc_start = -1
                for dc_idx in reversed(death_crosses):
                    if dc_idx < current_gc_signal_point:
                        current_phase_dc_start = dc_idx
                        break
                if current_phase_dc_start == -1:
                    continue

                hist_current_phase = hist[
                    current_phase_dc_start : current_gc_signal_point + 1
                ]
                metric_current_phase = self._calculate_metric(
                    hist_current_phase, comparison_metric, True
                )

                prev_phase_gc_end = -1
                for prev_gc_candidate_idx in reversed(golden_crosses):
                    if prev_gc_candidate_idx < current_phase_dc_start:
                        prev_phase_gc_end = prev_gc_candidate_idx
                        break
                if prev_phase_gc_end == -1:
                    continue

                prev_phase_dc_start = -1
                for prev_dc_candidate_idx in reversed(death_crosses):
                    if prev_dc_candidate_idx < prev_phase_gc_end:
                        prev_phase_dc_start = prev_dc_candidate_idx
                        break
                if prev_phase_dc_start == -1:
                    continue

                hist_prev_phase = hist[prev_phase_dc_start : prev_phase_gc_end + 1]
                metric_prev_phase = self._calculate_metric(
                    hist_prev_phase, comparison_metric, True
                )

                if not np.isnan(metric_current_phase) and not np.isnan(
                    metric_prev_phase
                ):
                    if metric_current_phase > metric_prev_phase:
                        result.iloc[current_gc_signal_point] = True

        else:  # Bearish Divergence
            for current_dc_signal_point in death_crosses:
                current_phase_gc_start = -1
                for gc_idx in reversed(golden_crosses):
                    if gc_idx < current_dc_signal_point:
                        current_phase_gc_start = gc_idx
                        break
                if current_phase_gc_start == -1:
                    continue

                hist_current_phase = hist[
                    current_phase_gc_start : current_dc_signal_point + 1
                ]
                metric_current_phase = self._calculate_metric(
                    hist_current_phase, comparison_metric, False
                )

                prev_phase_dc_end = -1
                for prev_dc_candidate_idx in reversed(death_crosses):
                    if prev_dc_candidate_idx < current_phase_gc_start:
                        prev_phase_dc_end = prev_dc_candidate_idx
                        break
                if prev_phase_dc_end == -1:
                    continue

                prev_phase_gc_start = -1
                for prev_gc_candidate_idx in reversed(golden_crosses):
                    if prev_gc_candidate_idx < prev_phase_dc_end:
                        prev_phase_gc_start = prev_gc_candidate_idx
                        break
                if prev_phase_gc_start == -1:
                    continue

                hist_prev_phase = hist[prev_phase_gc_start : prev_phase_dc_end + 1]
                metric_prev_phase = self._calculate_metric(
                    hist_prev_phase, comparison_metric, False
                )

                if not np.isnan(metric_current_phase) and not np.isnan(
                    metric_prev_phase
                ):
                    if metric_current_phase < metric_prev_phase:
                        result.iloc[current_dc_signal_point] = True
        return result


if __name__ == "__main__":
    # import time

    print("\n--- MA Cross策略回测 ---")
    ma_cross = MaCross(
        symbol="BTC-USD",
        start="2023-01-01 UTC",
        end="2025-06-01 UTC",
        fast_window=14,
        slow_window=21,
    )

    pf_ma = ma_cross.run()
    print(pf_ma.stats())
    # ma_cross.print_order_details(pf_ma)
    trades = pf_ma.trades.records_readable
    trades_subset = trades[
        [
            "Entry Timestamp",
            "Avg Entry Price",
            "Exit Timestamp",
            "Avg Exit Price",
            "PnL",
            "Return",
            "Direction",
        ]
    ]
    print(trades_subset)

    # print("\n--- 海龟策略回测 ---")
    # turtle = TurtleStrategy(
    #     symbol="BTC-USD",
    #     start="2023-01-01 UTC",
    #     end="2025-06-01 UTC",
    #     entry_window=20,
    #     exit_window=10,
    #     atr_window=20,
    #     risk_pct=0.01,
    # )
    # pf_turtle = turtle.run()
    # print(pf_turtle.stats())
    # trades_turtle = pf_turtle.trades.records_readable[
    #     [
    #         "Entry Timestamp",
    #         "Avg Entry Price",
    #         "Exit Timestamp",
    #         "Avg Exit Price",
    #         "PnL",
    #         "Return",
    #         "Direction",
    #     ]
    # ]
    # print(trades_turtle)

    # pf_ma.plot().show()

    # # 运行MACD底背离策略
    # print("\n--- MACD底背离策略回测 ---")
    # start_time_macd = time.time()
    # macd_div = MacdBullishDivergence(
    #     symbol="BTC-USD",
    #     start="2023-01-01 UTC",
    #     end="2025-06-01 UTC",
    #     comparison_metric="min_max",  # Or "area", "average"
    # )  # 明确传递日期
    # print(f"loading data took {time.time() - start_time_macd:.4f} seconds")
    # pf_macd = macd_div.run()
    # print(pf_macd.stats())
    # print("\n--- MACD策略交易订单 ---")
    # print(pf_macd.orders.records)

    # macd_div.print_order_details(pf_macd)
    # pf_macd.plot().show()

    # end_time_macd = time.time()
    # print(f"MACD底背离策略执行时间: {end_time_macd - start_time_macd:.4f} seconds")
