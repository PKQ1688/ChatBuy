from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table

from ..backtest.engine import BacktestEngine
from ..data.fetcher import DataFetcher
from ..data.processor import DataProcessor
from ..nlp.strategy_parser import StrategyParser
from ..strategies.strategy_factory import StrategyFactory


class ChatBuyCLI:
    """Main CLI interface for ChatBuy."""
    
    def __init__(self):
        self.console = Console()
        self.strategy_parser = StrategyParser()
        self.strategy_factory = StrategyFactory()
        self.backtest_engine = BacktestEngine()
        self.data_fetcher = DataFetcher()
        self.data_processor = DataProcessor()
    
    def run(self):
        """Main CLI loop."""
        self.console.print(Panel.fit(
            "[bold cyan]🤖 Welcome to ChatBuy[/bold cyan]\n"
            "[yellow]Interactive Quantitative Trading System[/yellow]\n\n"
            "[green]Type 'help' for commands or 'quit' to exit[/green]",
            title="ChatBuy"
        ))
        
        while True:
            try:
                # Get user input
                user_input = Prompt.ask("\n[cyan]💬 Describe your trading strategy[/cyan]")
                
                if user_input.lower() in ["quit", "exit", "q"]:
                    self.console.print("[yellow]👋 Goodbye![/yellow]")
                    break
                
                if user_input.lower() == "help":
                    self.show_help()
                    continue
                
                if user_input.lower() == "examples":
                    self.show_examples()
                    continue
                
                # Process the strategy request
                self.process_strategy_request(user_input)
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]👋 Goodbye![/yellow]")
                break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
    
    def process_strategy_request(self, user_input: str):
        """Process a single strategy request."""
        # Parse strategy
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Parsing strategy...", total=None)
            
            strategy_config = self.strategy_parser.parse(user_input)
            
            if strategy_config is None:
                self.console.print("[red]❌ Could not understand your strategy description[/red]")
                self.show_examples()
                return
            
            progress.update(task, description="Creating strategy...")
            
            # Create strategy
            strategy = self.strategy_factory.create_strategy(
                strategy_config["strategy_type"],
                strategy_config["parameters"]
            )
            
            if strategy is None:
                self.console.print("[red]❌ Could not create strategy[/red]")
                return
            
            progress.update(task, description="Fetching data...")
            
            # Fetch data (default to BTC for demo)
            data = self.data_fetcher.fetch_yfinance("BTC-USD")
            
            if data is None:
                self.console.print("[red]❌ Could not fetch data[/red]")
                return
            
            progress.update(task, description="Processing data...")
            
            # Clean data
            data = self.data_processor.clean_data(data)
            
            if not self.data_processor.validate_data(data):
                self.console.print("[red]❌ Invalid data[/red]")
                return
            
            progress.update(task, description="Running backtest...")
            
            # Run backtest
            results = self.backtest_engine.run_strategy_backtest(strategy, data)
            
            if results is None:
                self.console.print("[red]❌ Backtest failed[/red]")
                return
            
            # Show results
            self.backtest_engine.print_full_results(results)
    
    def show_help(self):
        """Show help information."""
        table = Table(title="Available Commands")
        table.add_column("Command", style="cyan")
        table.add_column("Description", style="white")
        
        table.add_row("help", "Show this help message")
        table.add_row("examples", "Show example strategy descriptions")
        table.add_row("quit/exit/q", "Exit the application")
        
        self.console.print(table)
    
    def show_examples(self):
        """Show example strategy descriptions."""
        examples = [
            "双均线金叉买入，20日均线和50日均线",
            "快线10日，慢线30日，金叉买入死叉卖出",
            "短期均线交叉长期均线，快线20慢线60",
            "moving average crossover with 10 and 30 periods",
        ]
        
        self.console.print(Panel.fit(
            "[bold cyan]Example Strategy Descriptions:[/bold cyan]\n\n" +
            "\n".join(f"[green]• {example}[/green]" for example in examples),
            title="Examples"
        ))