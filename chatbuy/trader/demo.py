from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import GOOG, SMA


class SmaCross(Strategy):
    """A strategy that uses two Simple Moving Averages (SMA) to generate buy and sell signals.

    Methods:
    -------
    init():
        Initializes the SMA indicators.
    next():
        Executes buy or sell orders based on SMA crossover signals.
    """

    def init(self):
        price = self.data.Close
        self.ma1 = self.I(SMA, price, 10)
        self.ma2 = self.I(SMA, price, 20)

    def next(self):
        if crossover(self.ma1, self.ma2):
            self.buy()
        elif crossover(self.ma2, self.ma1):
            self.sell()


bt = Backtest(GOOG, SmaCross, commission=0.002, exclusive_orders=True)
stats = bt.run()
# bt.plot()
print(stats)
