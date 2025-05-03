import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover


class BollingerRsiStrategy(Strategy):
    """Trading strategy using only Bollinger Bands and RSI.

    Indicators are calculated with talipp, crossovers use Backtest built-in methods.
    """

    def init(self):
        # 直接从数据中读取已存在的指标列
        self.bb_upper = self.data.bb_upper
        self.bb_mid = self.data.bb_middle
        self.bb_lower = self.data.bb_lower
        self.rsi = self.data.rsi

    def next(self):
        # Buy: Close crosses above BB lower band and RSI crosses above 30
        # Sell: Close crosses below BB upper band and RSI crosses below 70
        if crossover(self.data.Close, self.bb_lower) and crossover(self.rsi, 30):
            self.buy()
        elif crossover(self.bb_upper, self.data.Close) and crossover(70, self.rsi):
            self.sell()


if __name__ == "__main__":
    # Read raw data
    df = pd.read_csv(
        "data/BTC_USDT_1d_with_indicators.csv", index_col=0, parse_dates=True
    )
    # 只保留所有指标和价格都非 NaN 的数据
    df = df.dropna(subset=["Close", "bb_upper", "bb_middle", "bb_lower", "rsi"])

    bt = Backtest(
        df,
        BollingerRsiStrategy,
        commission=0.002,
        exclusive_orders=True,
        cash=1_000_000,
    )
    stats = bt.run()
    print(stats)
