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
        slippage: float = 0.001
    ) -> dict[str, Any] | None:
        """Run backtest for a given strategy."""
        try:
            # Validate data
            if not self._validate_data(data):
                return None
            
            # Generate signals
            self.console.print(f"[yellow]Generating signals for {strategy.name}...[/yellow]")
            signals = strategy.generate_signals(data)
            
            # Run backtest
            self.console.print("[yellow]Running backtest...[/yellow]")
            results = self.vbt_wrapper.run_backtest(
                data=data,
                signals=signals,
                initial_cash=initial_cash,
                fees=fees,
                slippage=slippage
            )
            
            # Add strategy info to results
            results["strategy"] = strategy.get_info()
            
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
            self.console.print(f"[red]Error: Missing required columns: {missing_columns}[/red]")
            return False
        
        return True
    
    def print_strategy_info(self, strategy: BaseStrategy):
        """Print strategy information."""
        info = strategy.get_info()
        
        self.console.print(Panel.fit(
            f"[bold cyan]Strategy: {info['name']}[/bold cyan]\n"
            f"[yellow]Description: {info['description']}[/yellow]\n"
            f"[green]Parameters: {info['parameters']}[/green]",
            title="Strategy Information"
        ))
    
    def print_full_results(self, results: dict[str, Any]):
        """Print complete backtest results."""
        if results is None:
            return
        
        # Print strategy info
        strategy_info = results["strategy"]
        self.console.print(Panel.fit(
            f"[bold cyan]Strategy: {strategy_info['name']}[/bold cyan]\n"
            f"[yellow]Description: {strategy_info['description']}[/yellow]\n"
            f"[green]Parameters: {strategy_info['parameters']}[/green]",
            title="Strategy Information"
        ))
        
        # Print backtest results
        self.vbt_wrapper.print_results(results)