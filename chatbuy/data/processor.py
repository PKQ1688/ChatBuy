from typing import Any

import pandas as pd
from rich.console import Console


class DataProcessor:
    """Data preprocessing and validation."""
    
    def __init__(self):
        self.console = Console()
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate data for backtesting."""
        if data is None or len(data) == 0:
            self.console.print("[red]Error: No data provided[/red]")
            return False
        
        # Check required columns
        required_columns = ["Close"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            self.console.print(f"[red]Error: Missing required columns: {missing_columns}[/red]")
            return False
        
        # Check for missing values
        if data["Close"].isnull().any():
            self.console.print("[yellow]Warning: Missing values found in Close prices[/yellow]")
        
        return True
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data for backtesting."""
        df = data.copy()
        
        # Handle missing values
        df = df.dropna(subset=["Close"])
        
        # Ensure numeric columns
        numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Sort by date if date column exists
        date_columns = ["Date", "date", "DATE", "Datetime", "datetime"]
        for col in date_columns:
            if col in df.columns:
                df = df.sort_values(col)
                break
        
        # Reset index
        df = df.reset_index(drop=True)
        
        self.console.print(f"[green]Data cleaned: {len(df)} rows remaining[/green]")
        return df
    
    def get_data_info(self, data: pd.DataFrame) -> dict[str, Any]:
        """Get basic information about the data."""
        if data is None or len(data) == 0:
            return {}
        
        info = {
            "rows": len(data),
            "columns": list(data.columns),
            "date_range": None,
            "missing_values": data.isnull().sum().to_dict()
        }
        
        # Get date range
        date_columns = ["Date", "date", "DATE", "Datetime", "datetime"]
        for col in date_columns:
            if col in data.columns:
                info["date_range"] = {
                    "start": data[col].min(),
                    "end": data[col].max()
                }
                break
        
        return info
    
    def print_data_info(self, data: pd.DataFrame):
        """Print data information."""
        info = self.get_data_info(data)
        
        if not info:
            self.console.print("[red]No data information available[/red]")
            return
        
        self.console.print("[bold cyan]Data Information:[/bold cyan]")
        self.console.print(f"Rows: {info['rows']}")
        self.console.print(f"Columns: {info['columns']}")
        
        if info["date_range"]:
            self.console.print(f"Date Range: {info['date_range']['start']} to {info['date_range']['end']}")
        
        # Show missing values
        missing_cols = {k: v for k, v in info["missing_values"].items() if v > 0}
        if missing_cols:
            self.console.print(f"[yellow]Missing values: {missing_cols}[/yellow]")
        else:
            self.console.print("[green]No missing values[/green]")