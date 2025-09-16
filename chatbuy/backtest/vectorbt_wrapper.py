from typing import Any

import pandas as pd
import vectorbt as vbt
from rich.console import Console
from rich.table import Table


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
    ) -> dict[str, Any]:
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

    def _calculate_stats(
        self, portfolio: Any, data: pd.DataFrame
    ) -> dict[str, Any]:
        """Calculate performance statistics.

        Also computes the instrument's own (buy-and-hold) return across the
        backtest period for quick comparison against the strategy.
        """
        stats: dict[str, Any] = {}

        try:
            # Basic stats
            stats["total_return"] = portfolio.total_return()
            stats["annualized_return"] = portfolio.annualized_return()
            stats["sharpe_ratio"] = portfolio.sharpe_ratio()
            stats["max_drawdown"] = portfolio.max_drawdown()
            stats["win_rate"] = portfolio.trades.win_rate()
            stats["total_trades"] = len(portfolio.trades)
            stats["profit_factor"] = portfolio.trades.profit_factor()

            # Handle annualized return calculation manually if needed
            if stats["annualized_return"] is None or pd.isna(
                stats["annualized_return"]
            ):
                # Manual calculation based on total return and time period
                total_days = len(portfolio.returns())
                if total_days > 0:
                    portfolio_years = total_days / 252  # Trading days in a year
                    if portfolio_years > 0:
                        stats["annualized_return"] = (1 + stats["total_return"]) ** (
                            1 / portfolio_years
                        ) - 1

            # Compute instrument (buy-and-hold) baseline
            try:
                # Ensure close_series is always a pandas Series before dropna/iloc
                close_raw = pd.to_numeric(data["Close"], errors="coerce")
                if not isinstance(close_raw, pd.Series):
                    close_series = pd.Series(close_raw).dropna()
                else:
                    close_series = close_raw.dropna()
                instrument_total_return: float | None = None
                instrument_annualized_return: float | None = None
                if len(close_series) >= 2:
                    instrument_total_return = float(
                        close_series.iloc[-1] / close_series.iloc[0] - 1
                    )

                    # Determine period length in years
                    date_col = next(
                        (
                            c
                            for c in [
                                "Date",
                                "Datetime",
                                "date",
                                "datetime",
                                "DATE",
                            ]
                            if c in data.columns
                        ),
                        None,
                    )
                    years: float | None = None
                    if date_col is not None:
                        try:
                            start_ts = pd.to_datetime(data[date_col].min())
                            end_ts = pd.to_datetime(data[date_col].max())
                            days = (end_ts - start_ts).days
                            if days > 0:
                                years = days / 365.25
                        except Exception:
                            years = None
                    if years is None:
                        # Fallback using trading days
                        years = (
                            len(close_series) / 252.0
                            if len(close_series) > 0
                            else None
                        )

                    if (
                        years is not None
                        and years > 0
                        and instrument_total_return is not None
                    ):
                        instrument_annualized_return = (
                            1 + instrument_total_return
                        ) ** (1 / years) - 1

                stats["instrument_total_return"] = instrument_total_return
                stats["instrument_annualized_return"] = instrument_annualized_return
            except Exception:
                stats["instrument_total_return"] = None
                stats["instrument_annualized_return"] = None

            # Format for display
            stats["formatted"] = {
                "Total Return": f"{stats['total_return']:.2%}",
                "Annualized Return": f"{stats['annualized_return']:.2%}",
                "Sharpe Ratio": f"{stats['sharpe_ratio']:.2f}",
                "Max Drawdown": f"{stats['max_drawdown']:.2%}",
                "Win Rate": f"{stats['win_rate']:.2%}",
                "Total Trades": f"{stats['total_trades']}",
                "Profit Factor": f"{stats['profit_factor']:.2f}",
            }

            # Add instrument returns to formatted block when available
            if stats["instrument_total_return"] is not None:
                stats["formatted"]["Instrument Return"] = (
                    f"{stats['instrument_total_return']:.2%}"
                )
            if stats["instrument_annualized_return"] is not None:
                stats["formatted"]["Instrument Ann. Return"] = (
                    f"{stats['instrument_annualized_return']:.2%}"
                )

            # Excess Return (strategy - instrument)
            if (
                stats.get("total_return") is not None
                and stats.get("instrument_total_return") is not None
            ):
                excess = stats["total_return"] - stats["instrument_total_return"]
                stats["excess_return"] = excess
                stats["formatted"]["Excess Return"] = f"{excess:.2%}"

        except Exception as e:
            self.console.print(f"[red]Error calculating stats: {e}[/red]")
            stats["error"] = str(e)

        return stats

    def print_results(self, results: dict[str, Any]):
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
        self, results: dict[str, Any], show: bool = True
    ) -> Any | None:
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
