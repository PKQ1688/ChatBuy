from typing import Any

import pandas as pd
from rich.console import Console
from rich.panel import Panel

from ..strategies.base_strategy import BaseStrategy
from .vectorbt_wrapper import VectorbtWrapper


class BacktestEngine:
    """Main backtest engine."""

    def __init__(self):
        self.vbt_wrapper = VectorbtWrapper()
        self.console = Console()

    def run_strategy_backtest(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        initial_cash: float = 10000.0,
        fees: float = 0.001,
        slippage: float = 0.001,
    ) -> dict[str, Any] | None:
        """Run backtest for a given strategy."""
        try:
            # Validate data
            if not self._validate_data(data):
                return None

            # Generate signals
            self.console.print(
                f"[yellow]Generating signals for {strategy.name}...[/yellow]"
            )
            signals = strategy.generate_signals(data)

            # Run backtest
            self.console.print("[yellow]Running backtest...[/yellow]")
            results = self.vbt_wrapper.run_backtest(
                data=data,
                signals=signals,
                initial_cash=initial_cash,
                fees=fees,
                slippage=slippage,
            )

            # Add strategy info to results
            results["strategy"] = strategy.get_info()

            # Attach metadata: instrument and period
            meta: dict[str, Any] = {
                "symbol": data.attrs.get("symbol"),
                "instrument_name": data.attrs.get(
                    "instrument_name", data.attrs.get("symbol")
                ),
                "source": data.attrs.get("source"),
                "interval": data.attrs.get("interval"),
            }

            # Determine trading period from data (fallback if attrs missing)
            date_cols = ["Date", "Datetime", "date", "datetime", "DATE"]
            date_col = next((c for c in date_cols if c in data.columns), None)
            start_iso: str | None = None
            end_iso: str | None = None
            if data.attrs.get("date_range"):
                dr = data.attrs.get("date_range")
                start_iso = dr.get("start") if isinstance(dr, dict) else None
                end_iso = dr.get("end") if isinstance(dr, dict) else None
            elif date_col is not None:
                try:
                    start_iso = pd.to_datetime(data[date_col].min()).isoformat()
                    end_iso = pd.to_datetime(data[date_col].max()).isoformat()
                except Exception:
                    start_iso = None
                    end_iso = None

            meta["period"] = {"start": start_iso, "end": end_iso}
            results["meta"] = meta

            return results

        except Exception as e:
            self.console.print(f"[red]Error running backtest: {e}[/red]")
            return None

    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data."""
        if data is None or len(data) == 0:
            self.console.print("[red]Error: No data provided[/red]")
            return False

        required_columns = ["Close"]
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            self.console.print(
                f"[red]Error: Missing required columns: {missing_columns}[/red]"
            )
            return False

        return True

    def print_strategy_info(self, strategy: BaseStrategy) -> None:
        """Print strategy information."""
        info = strategy.get_info()

        self.console.print(
            Panel.fit(
                f"[bold cyan]Strategy: {info['name']}[/bold cyan]\n"
                f"[yellow]Description: {info['description']}[/yellow]\n"
                f"[green]Parameters: {info['parameters']}[/green]",
                title="Strategy Information",
            )
        )

    def print_full_results(self, results: dict[str, Any]) -> None:
        """Print complete backtest results."""
        if results is None:
            return

        # Print strategy info
        strategy_info = results["strategy"]
        self.console.print(
            Panel.fit(
                f"[bold cyan]Strategy: {strategy_info['name']}[/bold cyan]\n"
                f"[yellow]Description: {strategy_info['description']}[/yellow]\n"
                f"[green]Parameters: {strategy_info['parameters']}[/green]",
                title="Strategy Information",
            )
        )

        # Print instrument and trading period
        meta = results.get("meta", {})
        symbol = meta.get("symbol") or "—"
        inst_name = meta.get("instrument_name") or symbol
        interval = meta.get("interval") or "—"
        period = meta.get("period") or {}
        start = period.get("start") or "—"
        end = period.get("end") or "—"

        self.console.print(
            Panel.fit(
                f"[bold cyan]Instrument:[/bold cyan] {inst_name} ({symbol})\n"
                f"[yellow]Trading Period:[/yellow] {start} → {end}\n"
                f"[green]Data Frequency:[/green] {interval}",
                title="Data",
            )
        )

        # Print backtest results
        self.vbt_wrapper.print_results(results)
