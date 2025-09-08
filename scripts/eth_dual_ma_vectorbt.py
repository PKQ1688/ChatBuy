#!/usr/bin/env python3
"""ETH dual moving-average backtest demo using vectorbt.

Features
- Data source: CSV or Yahoo Finance (`yfinance`)
- Single-parameter run or grid search over MA windows
- Summary stats printout; optional interactive plot

Examples:
- CSV (provide your own OHLCV CSV with a Close column):
  python scripts/eth_dual_ma_vectorbt.py --source csv --csv-path data/ETH-USD.csv --fast 20 --slow 50

- Yahoo Finance (requires yfinance and network):
  python scripts/eth_dual_ma_vectorbt.py --source yfinance --symbol ETH-USD --start 2020-01-01 --interval 1d

Grid search example:
  python scripts/eth_dual_ma_vectorbt.py --source yfinance --grid \
    --fast-list 5,10,20,50 --slow-list 100,150,200 --top 5
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _parse_list(csv: str) -> list[int]:
    return [int(x.strip()) for x in csv.split(",") if x.strip()]


def load_prices_from_csv(csv_path: str) -> pd.Series:
    """Load price data from CSV file.
    
    Args:
        csv_path: Path to CSV file containing price data
        
    Returns:
        pd.Series: Price series with datetime index
    """
    df = pd.read_csv(csv_path)
    # Try to be flexible about column names
    date_col = None
    for c in ["Date", "date", "Datetime", "datetime", "Timestamp", "timestamp"]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        raise ValueError("CSV must include a date/datetime column")

    price_col = None
    for c in ["Close", "close", "Adj Close", "adj_close", "AdjClose"]:
        if c in df.columns:
            price_col = c
            break
    if price_col is None:
        raise ValueError("CSV must include a Close/Adj Close column")

    s = pd.Series(
        df[price_col].values, index=pd.to_datetime(df[date_col], utc=False)
    ).sort_index().astype(float)
    s.name = "Close"
    # Try to set a daily frequency if possible (not required by vectorbt)
    try:
        s = s.asfreq("1D")
    except Exception:
        pass
    return s


def load_prices_from_yfinance(symbol: str, start: str | None, end: str | None, interval: str) -> pd.Series:
    """Load price data from Yahoo Finance.
    
    Args:
        symbol: Stock symbol
        start: Start date
        end: End date
        interval: Data interval
        
    Returns:
        pd.Series: Price series with datetime index
    """
    try:
        import yfinance as yf
    except Exception as e:  # pragma: no cover - optional dependency
        raise RuntimeError("yfinance is required for --source yfinance. Install with: pip install yfinance") from e

    df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    if df.empty:  # type: ignore
        raise RuntimeError("No data returned from yfinance. Check symbol/interval/date range.")
    s = df["Close"].dropna()  # type: ignore
    # yfinance may return a DataFrame when using multi-index columns
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s = s.astype(float)
    s.name = "Close"
    return s


# Note: synthetic data generation was removed per repo policy to prefer
# real data sources (yfinance by default) or user-provided CSV.


@dataclass
class BacktestConfig:
    """Configuration for backtest parameters."""
    fast: int
    slow: int
    fees: float
    cash: float
    plot: bool


def run_single(price: pd.Series, cfg: BacktestConfig) -> None:
    """Run a single backtest with given configuration.
    
    Args:
        price: Price series
        cfg: Backtest configuration
    """
    import vectorbt as vbt

    if cfg.fast >= cfg.slow:
        print("Warning: fast window should be < slow window for a classic dual-MA crossover.")

    fast_ma = vbt.MA.run(price, window=cfg.fast)  # type: ignore
    slow_ma = vbt.MA.run(price, window=cfg.slow)  # type: ignore
    entries = fast_ma.ma_crossed_above(slow_ma)  # type: ignore
    exits = fast_ma.ma_crossed_below(slow_ma)  # type: ignore

    pf = vbt.Portfolio.from_signals(
        price,
        entries=entries,
        exits=exits,
        init_cash=cfg.cash,
        fees=cfg.fees,
        freq="1D",
        slippage=0.0,
    )

    print("=== Summary (single run) ===")
    try:
        # Newer vectorbt returns a Series for stats
        stats = pf.stats()  # type: ignore
        print(stats.to_string() if hasattr(stats, "to_string") else str(stats))  # type: ignore
    except Exception:
        # Fallback: print a few key metrics
        print({
            "Start": price.index.min(),
            "End": price.index.max(),
            "Final Value": float(pf.final_value()),  # type: ignore
            "Total Return [%]": float((pf.total_return() * 100).round(4)),  # type: ignore
            "Sharpe": float(pf.sharpe_ratio().round(4)),  # type: ignore
            "Max Drawdown [%]": float((pf.max_drawdown() * 100).round(4)),  # type: ignore
            "Trades": int(pf.trades.count()) if hasattr(pf, "trades") else np.nan,  # type: ignore
        })

    if cfg.plot:
        try:
            fig = pf.plot()  # type: ignore
            # Overlay MAs
            import plotly.graph_objects as go

            if fig is not None:
                fig.add_trace(go.Scatter(x=price.index, y=fast_ma.ma, name=f"MA{cfg.fast}", line=dict(width=1)))  # type: ignore
                fig.add_trace(go.Scatter(x=price.index, y=slow_ma.ma, name=f"MA{cfg.slow}", line=dict(width=1)))  # type: ignore
                fig.show()
        except Exception as e:
            print(f"Plotting failed: {e}")


def run_grid(price: pd.Series, fast_list: Iterable[int], slow_list: Iterable[int], fees: float, cash: float, top: int, plot: bool) -> None:
    """Run grid search over multiple parameter combinations.
    
    Args:
        price: Price series
        fast_list: List of fast window values
        slow_list: List of slow window values
        fees: Trading fees
        cash: Initial cash
        top: Number of top results to show
        plot: Whether to show plots
    """
    import vectorbt as vbt

    fast_windows = list(fast_list)
    slow_windows = list(slow_list)

    # Keep only fast < slow to avoid trivial/self-overlap combos
    combos: list[tuple[int, int]] = [(f, s) for f in fast_windows for s in slow_windows if f < s]
    if not combos:
        raise ValueError("No valid (fast, slow) pairs where fast < slow")

    # Run MA computations broadcasted over window lists
    fast_ma = vbt.MA.run(price, window=fast_windows)  # type: ignore
    slow_ma = vbt.MA.run(price, window=slow_windows)  # type: ignore

    # Broadcast to 2D by aligning indexes; vectorbt will create a 2D grid via cross comparison
    entries = fast_ma.ma_crossed_above(slow_ma)  # type: ignore
    exits = fast_ma.ma_crossed_below(slow_ma)  # type: ignore

    pf = vbt.Portfolio.from_signals(
        price,
        entries=entries,
        exits=exits,
        init_cash=cash,
        fees=fees,
        freq="1D",
    )

    final_val = pf.final_value()  # type: ignore
    # final_val is a 2D DataFrame indexed by fast window (rows) and slow window (cols)
    results = final_val.stack().rename("final_value").to_frame()
    results["total_return"] = (results["final_value"] / cash) - 1.0

    # Attach Sharpe if available
    try:
        sharpe = pf.sharpe_ratio()  # type: ignore
        results["sharpe"] = sharpe.stack()
    except Exception:
        pass

    results = results.sort_values("final_value", ascending=False)
    print("=== Top results (grid) ===")
    print(results.head(top))

    if plot:
        try:
            import plotly.express as px

            heatmap_df = final_val.copy()
            fig = px.imshow(
                heatmap_df,
                labels=dict(x="Slow Window", y="Fast Window", color="Final Value"),
                x=heatmap_df.columns,
                y=heatmap_df.index,
                aspect="auto",
                title="Final Value Heatmap",
            )
            fig.show()
        except Exception as e:
            print(f"Heatmap plotting failed: {e}")


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the script.
    
    Args:
        argv: Command line arguments
        
    Returns:
        int: Exit code
    """
    p = argparse.ArgumentParser(description="ETH dual MA backtest demo (vectorbt)")
    src = p.add_argument_group("Data Source")
    src.add_argument("--source", choices=["csv", "yfinance"], default="yfinance")
    src.add_argument("--csv-path", type=str, default="data/ETH-USD.csv", help="CSV path with Date and Close columns")
    src.add_argument("--symbol", type=str, default="ETH-USD", help="Ticker/symbol for yfinance")
    src.add_argument("--start", type=str, default="2018-01-01")
    src.add_argument("--end", type=str, default=None)
    src.add_argument("--interval", type=str, default="1d", help="yfinance interval, e.g. 1d/1h/5m")

    bt = p.add_argument_group("Backtest")
    bt.add_argument("--fast", type=int, default=20)
    bt.add_argument("--slow", type=int, default=50)
    bt.add_argument("--fees", type=float, default=0.0007, help="Proportional fee per trade, e.g. 0.0007=7bps")
    bt.add_argument("--cash", type=float, default=10_000)
    bt.add_argument("--plot", action="store_true")

    grid = p.add_argument_group("Grid Search")
    grid.add_argument("--grid", action="store_true", help="Run a grid search over fast/slow lists")
    grid.add_argument("--fast-list", type=str, default="5,10,20,50")
    grid.add_argument("--slow-list", type=str, default="60,100,150,200")
    grid.add_argument("--top", type=int, default=10, help="Show top-N combos by final value")

    args = p.parse_args(argv)

    # Load data
    if args.source == "csv":
        price = load_prices_from_csv(args.csv_path)
    elif args.source == "yfinance":
        price = load_prices_from_yfinance(args.symbol, args.start, args.end, args.interval)
    else:  # pragma: no cover - defensive
        raise ValueError("Unsupported source. Choose from: csv, yfinance.")

    # Ensure positive prices
    price = price.replace([np.inf, -np.inf], np.nan).dropna()
    # Robust check for non-positive values for both Series/DataFrame
    if np.any(price.to_numpy() <= 0):
        raise ValueError("Prices must be positive for log/ratio calculations.")

    if args.grid:
        fast_list = _parse_list(args.fast_list)
        slow_list = _parse_list(args.slow_list)
        run_grid(price, fast_list, slow_list, args.fees, args.cash, args.top, args.plot)
    else:
        cfg = BacktestConfig(fast=args.fast, slow=args.slow, fees=args.fees, cash=args.cash, plot=args.plot)
        run_single(price, cfg)

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
