from typing import Any

import pandas as pd
import vectorbt as vbt
from rich.console import Console
from rich.table import Table

# Type aliases for vectorbt objects and stats
type StatsValue = float | int | str | dict[str, str] | None
type StatsDict = dict[str, StatsValue]
type BacktestResults = dict[str, Any]


class VectorbtWrapper:
    """Wrapper for vectorbt backtesting functionality."""

    def __init__(self):
        self.console = Console()

    def run_backtest(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        initial_cash: float = 10000.0,
        fees: float = 0.001,
        slippage: float = 0.001,
    ) -> BacktestResults:
        """Run backtest using vectorbt."""
        # Create portfolio from signals with frequency
        portfolio = vbt.Portfolio.from_signals(
            close=data["Close"],
            entries=signals == 1,
            exits=signals == -1,
            init_cash=initial_cash,
            fees=fees,
            slippage=slippage,
            direction="longonly",  # Only long positions for now
            freq="1D",  # Daily frequency for Yahoo Finance data
        )

        # Calculate performance metrics (including instrument baseline)
        stats = self._calculate_stats(portfolio, data)

        return {
            "portfolio": portfolio,
            "stats": stats,
            "equity_curve": portfolio.value(),
            "returns": portfolio.returns(),
            "trades": portfolio.trades,
        }

    def _calculate_stats(self, portfolio: Any, data: pd.DataFrame) -> StatsDict:
        """Calculate performance statistics by coordinating helper methods.

        Also computes the instrument's own (buy-and-hold) return across the
        backtest period for quick comparison against the strategy.
        """
        stats: StatsDict = {}
        try:
            # 1. Get base stats from vectorbt portfolio
            self._calculate_base_portfolio_stats(portfolio, stats)

            # 2. Manually calculate annualized return if missing
            self._calculate_manual_annualized_return(portfolio, stats)

            # 3. Compute instrument (buy-and-hold) baseline
            instrument_stats = self._calculate_instrument_baseline(data)
            stats.update(instrument_stats)

            # 4. Format for display and add excess return
            self._format_stats(stats)

        except Exception as e:
            self.console.print(f"[red]Error calculating stats: {e}[/red]")
            stats["error"] = str(e)

        return stats

    def _calculate_base_portfolio_stats(
        self, portfolio: Any, stats: StatsDict
    ) -> None:
        """Calculates basic stats from the portfolio object."""
        stats["total_return"] = portfolio.total_return()
        stats["annualized_return"] = portfolio.annualized_return()
        stats["sharpe_ratio"] = portfolio.sharpe_ratio()
        stats["max_drawdown"] = portfolio.max_drawdown()
        stats["win_rate"] = portfolio.trades.win_rate()
        stats["total_trades"] = len(portfolio.trades.records_readable)  # type: ignore[arg-type]
        stats["profit_factor"] = portfolio.trades.profit_factor()

    def _calculate_manual_annualized_return(
        self, portfolio: Any, stats: StatsDict
    ) -> None:
        """Manually calculates annualized return if it's missing."""
        annualized = stats.get("annualized_return")
        if annualized is not None and not pd.isna(annualized):  # type: ignore[arg-type]
            return

        total_days = len(portfolio.returns())
        if total_days <= 0:
            return

        portfolio_years = total_days / 252  # Trading days in a year
        total_return = stats.get("total_return")
        if portfolio_years > 0 and total_return is not None:
            stats["annualized_return"] = (1 + total_return) ** (  # type: ignore[operator]
                1 / portfolio_years
            ) - 1

    def _get_years_from_data(
        self, data: pd.DataFrame, close_series: pd.Series
    ) -> float | None:
        """Determines the period length in years from the dataframe."""
        date_col = next(
            (
                c
                for c in ["Date", "Datetime", "date", "datetime", "DATE"]
                if c in data.columns
            ),
            None,
        )
        if date_col is not None:
            try:
                start_ts = pd.to_datetime(data[date_col].min())
                end_ts = pd.to_datetime(data[date_col].max())
                days = (end_ts - start_ts).days
                if days > 0:
                    return days / 365.25
            except Exception:
                pass  # Fallback to trading days

        return len(close_series) / 252.0 if len(close_series) > 0 else None

    def _calculate_instrument_baseline(self, data: pd.DataFrame) -> StatsDict:
        """Computes the instrument's own (buy-and-hold) return."""
        instrument_stats: StatsDict = {
            "instrument_total_return": None,
            "instrument_annualized_return": None,
        }
        try:
            close_raw = pd.to_numeric(data["Close"], errors="coerce")
            close_series = (
                pd.Series(close_raw).dropna()
                if not isinstance(close_raw, pd.Series)
                else close_raw.dropna()
            )

            if len(close_series) < 2:
                return instrument_stats

            total_return = float(close_series.iloc[-1] / close_series.iloc[0] - 1)
            instrument_stats["instrument_total_return"] = total_return

            years = self._get_years_from_data(data, close_series)
            if years is not None and years > 0:
                annualized_return = (1 + total_return) ** (1 / years) - 1
                instrument_stats["instrument_annualized_return"] = annualized_return

        except Exception:
            # Silently fail, returning None for instrument stats
            pass

        return instrument_stats

    def _format_stats(self, stats: StatsDict) -> None:
        """Formats the calculated stats for display and adds excess return."""
        # Ensure keys exist before formatting
        for key in [
            "total_return",
            "annualized_return",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
            "profit_factor",
        ]:
            stats.setdefault(key, 0.0)
        stats.setdefault("total_trades", 0)

        stats["formatted"] = {  # type: ignore[assignment]
            "Total Return": f"{stats['total_return']:.2%}",  # type: ignore[str-format]
            "Annualized Return": f"{stats['annualized_return']:.2%}",  # type: ignore[str-format]
            "Sharpe Ratio": f"{stats['sharpe_ratio']:.2f}",  # type: ignore[str-format]
            "Max Drawdown": f"{stats['max_drawdown']:.2%}",  # type: ignore[str-format]
            "Win Rate": f"{stats['win_rate']:.2%}",  # type: ignore[str-format]
            "Total Trades": f"{stats['total_trades']}",
            "Profit Factor": f"{stats['profit_factor']:.2f}",  # type: ignore[str-format]
        }

        if stats.get("instrument_total_return") is not None:
            stats["formatted"]["Instrument Return"] = (  # type: ignore[index]
                f"{stats['instrument_total_return']:.2%}"  # type: ignore[str-format]
            )
        if stats.get("instrument_annualized_return") is not None:
            stats["formatted"]["Instrument Ann. Return"] = (  # type: ignore[index]
                f"{stats['instrument_annualized_return']:.2%}"  # type: ignore[str-format]
            )

        # Excess Return
        total_ret = stats.get("total_return")
        instrument_ret = stats.get("instrument_total_return")
        if total_ret is not None and instrument_ret is not None:
            excess = total_ret - instrument_ret  # type: ignore[operator]
            stats["excess_return"] = excess
            stats["formatted"]["Excess Return"] = f"{excess:.2%}"  # type: ignore[str-format]

    def print_results(self, results: BacktestResults):
        """Print backtest results in a formatted table."""
        if "error" in results["stats"]:
            self.console.print(f"[red]Error: {results['stats']['error']}[/red]")
            return

        # Create performance table
        table = Table(title="Backtest Results")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        for metric, value in results["stats"]["formatted"].items():
            table.add_row(metric, value)

        self.console.print(table)

    def plot_equity_curve(
        self, results: BacktestResults, show: bool = True
    ) -> Any:
        """Plot equity curve."""
        try:
            equity_curve = results["equity_curve"]

            # Create simple plot
            fig = equity_curve.vbt.plot(title="Portfolio Equity Curve")

            if show:
                fig.show()

            return fig

        except Exception as e:
            self.console.print(f"[red]Error plotting equity curve: {e}[/red]")
            return None
