
import pandas as pd
import yfinance as yf
from rich.console import Console


class DataFetcher:
    """Data fetching from various sources."""
    
    def __init__(self):
        self.console = Console()
    
    def fetch_yfinance(
        self,
        symbol: str,
        start_date: str = "2020-01-01",
        end_date: str = None,
        interval: str = "1d"
    ) -> pd.DataFrame | None:
        """Fetch data from Yahoo Finance."""
        try:
            self.console.print(f"[yellow]Fetching data for {symbol} from Yahoo Finance...[/yellow]")
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval
            )
            
            if len(data) == 0:
                self.console.print(f"[red]No data found for {symbol}[/red]")
                return None
            
            # Reset index to make Date a column
            data = data.reset_index()
            
            # Standardize column names
            data.columns = [col.title() for col in data.columns]
            
            self.console.print(f"[green]Successfully fetched {len(data)} rows of data[/green]")
            return data
            
        except Exception as e:
            self.console.print(f"[red]Error fetching data from Yahoo Finance: {e}[/red]")
            return None
    
    def fetch_csv(self, file_path: str) -> pd.DataFrame | None:
        """Fetch data from CSV file."""
        try:
            self.console.print(f"[yellow]Loading data from CSV: {file_path}[/yellow]")
            
            data = pd.read_csv(file_path)
            
            # Standardize column names
            data.columns = [col.title() for col in data.columns]
            
            # Try to parse date column
            date_columns = ["Date", "date", "DATE", "Datetime", "datetime"]
            for col in date_columns:
                if col in data.columns:
                    data[col] = pd.to_datetime(data[col])
                    break
            
            self.console.print(f"[green]Successfully loaded {len(data)} rows of data[/green]")
            return data
            
        except Exception as e:
            self.console.print(f"[red]Error loading CSV file: {e}[/red]")
            return None
    
    def get_available_sources(self) -> dict[str, str]:
        """Get available data sources."""
        return {
            "yfinance": "Yahoo Finance",
            "csv": "CSV File"
        }