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
        end_date: str | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame | None:
        """Fetch data from Yahoo Finance.

        Attaches basic metadata to ``DataFrame.attrs`` for downstream use:
        - ``symbol``: requested ticker symbol
        - ``instrument_name``: human-readable name if available; falls back to symbol
        - ``source``: data source identifier (``"yfinance"``)
        - ``interval``: requested bar interval (e.g., ``"1d"``)
        - ``date_range``: dict with ``start`` and ``end`` if determinable
        """
        try:
            self.console.print(
                f"[yellow]Fetching data for {symbol} from Yahoo Finance...[/yellow]"
            )

            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)

            if len(data) == 0:
                self.console.print(f"[red]No data found for {symbol}[/red]")
                return None

            # Reset index to make Date a column
            data = data.reset_index()

            # Standardize column names
            data.columns = [col.title() for col in data.columns]

            # Attach metadata for downstream consumers
            instrument_name = None
            try:
                # yfinance >=0.2 provides get_info; this may fail for some symbols
                info = ticker.get_info()  # type: ignore[attr-defined]
                if isinstance(info, dict):
                    instrument_name = (
                        info.get("longName")
                        or info.get("shortName")
                        or info.get("name")
                    )
            except Exception:
                # Best-effort; silently ignore metadata failures
                instrument_name = None

            # Compute date range if we can
            date_col = next(
                (
                    c
                    for c in ["Date", "Datetime", "date", "datetime", "DATE"]
                    if c in data.columns
                ),
                None,
            )
            date_range: dict[str, str] | None = None
            if date_col is not None:
                try:
                    start_val = pd.to_datetime(data[date_col].min()).isoformat()
                    end_val = pd.to_datetime(data[date_col].max()).isoformat()
                    date_range = {"start": start_val, "end": end_val}
                except Exception:
                    date_range = None

            data.attrs.update(
                {
                    "symbol": symbol,
                    "instrument_name": instrument_name or symbol,
                    "source": "yfinance",
                    "interval": interval,
                    "date_range": date_range,
                }
            )

            self.console.print(
                f"[green]Successfully fetched {len(data)} rows of data[/green]"
            )
            return data

        except Exception as e:
            self.console.print(
                f"[red]Error fetching data from Yahoo Finance: {e}[/red]"
            )
            return None

    def fetch_csv(self, file_path: str) -> pd.DataFrame | None:
        """Fetch data from CSV file.

        Attempts to set ``DataFrame.attrs`` with lightweight metadata:
        - ``symbol`` derived from filename (without extension)
        - ``instrument_name`` falls back to the same value
        - ``source`` is ``"csv"``
        - ``interval`` left as ``"unknown"``
        - ``date_range`` computed if a date-like column exists
        """
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

            # Attach basic metadata
            import os

            symbol_guess = os.path.splitext(os.path.basename(file_path))[0]

            # Compute date range if possible
            date_col = next(
                (
                    c
                    for c in ["Date", "Datetime", "date", "datetime", "DATE"]
                    if c in data.columns
                ),
                None,
            )
            date_range: dict[str, str] | None = None
            if date_col is not None:
                try:
                    start_val = pd.to_datetime(data[date_col].min()).isoformat()
                    end_val = pd.to_datetime(data[date_col].max()).isoformat()
                    date_range = {"start": start_val, "end": end_val}
                except Exception:
                    date_range = None

            data.attrs.update(
                {
                    "symbol": symbol_guess,
                    "instrument_name": symbol_guess,
                    "source": "csv",
                    "interval": "unknown",
                    "date_range": date_range,
                }
            )

            self.console.print(
                f"[green]Successfully loaded {len(data)} rows of data[/green]"
            )
            return data

        except Exception as e:
            self.console.print(f"[red]Error loading CSV file: {e}[/red]")
            return None

    def get_available_sources(self) -> dict[str, str]:
        """Get available data sources."""
        return {"yfinance": "Yahoo Finance", "csv": "CSV File"}
