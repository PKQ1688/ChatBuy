#!/usr/bin/env python3
"""Simple demo of ChatBuy system."""

from rich.console import Console

from chatbuy.backtest.engine import BacktestEngine
from chatbuy.data.fetcher import DataFetcher
from chatbuy.data.processor import DataProcessor
from chatbuy.nlp.strategy_parser import StrategyParser
from chatbuy.strategies.strategy_factory import StrategyFactory


def demo_ma_strategy():
    """Demo of moving average strategy."""
    console = Console()
    
    # Step 1: Parse natural language
    console.print("[bold cyan]Step 1: Parsing natural language...[/bold cyan]")
    parser = StrategyParser()
    strategy_desc = "双均线金叉买入，20日均线和50日均线"
    
    console.print(f"Input: {strategy_desc}")
    config = parser.parse(strategy_desc)
    
    if config:
        console.print(f"[green]✓ Strategy type: {config['strategy_type']}[/green]")
        console.print(f"[green]✓ Parameters: {config['parameters']}[/green]")
    else:
        console.print("[red]✗ Failed to parse strategy[/red]")
        return
    
    # Step 2: Create strategy
    console.print("\n[bold cyan]Step 2: Creating strategy...[/bold cyan]")
    factory = StrategyFactory()
    strategy = factory.create_strategy(config["strategy_type"], config["parameters"])
    
    if strategy:
        console.print(f"[green]✓ Strategy created: {strategy.name}[/green]")
    else:
        console.print("[red]✗ Failed to create strategy[/red]")
        return
    
    # Step 3: Fetch data
    console.print("\n[bold cyan]Step 3: Fetching data...[/bold cyan]")
    fetcher = DataFetcher()
    processor = DataProcessor()
    
    data = fetcher.fetch_yfinance("BTC-USD", start_date="2024-01-01", end_date="2024-06-01")
    
    if data is not None and len(data) > 0:
        console.print(f"[green]✓ Data fetched: {len(data)} rows[/green]")
        data = processor.clean_data(data)
    else:
        console.print("[red]✗ Failed to fetch data[/red]")
        return
    
    # Step 4: Run backtest
    console.print("\n[bold cyan]Step 4: Running backtest...[/bold cyan]")
    engine = BacktestEngine()
    results = engine.run_strategy_backtest(strategy, data)
    
    if results:
        console.print("[green]✓ Backtest completed![/green]")
        engine.print_full_results(results)
    else:
        console.print("[red]✗ Backtest failed[/red]")


if __name__ == "__main__":
    demo_ma_strategy()