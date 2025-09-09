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
        slippage: float = 0.001
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
            freq="1D"  # Daily frequency for Yahoo Finance data
        )
        
        # Calculate performance metrics
        stats = self._calculate_stats(portfolio)
        
        return {
            "portfolio": portfolio,
            "stats": stats,
            "equity_curve": portfolio.value(),
            "returns": portfolio.returns(),
            "trades": portfolio.trades
        }
    
    def _calculate_stats(self, portfolio) -> dict[str, Any]:
        """Calculate performance statistics."""
        stats = {}
        
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
            if stats["annualized_return"] is None or pd.isna(stats["annualized_return"]):
                # Manual calculation based on total return and time period
                total_days = len(portfolio.returns())
                if total_days > 0:
                    years = total_days / 252  # Trading days in a year
                    if years > 0:
                        stats["annualized_return"] = (1 + stats["total_return"]) ** (1 / years) - 1
            
            # Format for display
            stats["formatted"] = {
                "Total Return": f"{stats['total_return']:.2%}",
                "Annualized Return": f"{stats['annualized_return']:.2%}",
                "Sharpe Ratio": f"{stats['sharpe_ratio']:.2f}",
                "Max Drawdown": f"{stats['max_drawdown']:.2%}",
                "Win Rate": f"{stats['win_rate']:.2%}",
                "Total Trades": f"{stats['total_trades']}",
                "Profit Factor": f"{stats['profit_factor']:.2f}"
            }
            
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
    
    def plot_equity_curve(self, results: dict[str, Any], show: bool = True):
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