"""
投资组合管理器
"""

from typing import Dict, Optional
from datetime import datetime
import pandas as pd


class Portfolio:
    """投资组合管理器"""
    
    def __init__(self, initial_balance: float = 10000.0):
        """
        初始化投资组合
        
        Args:
            initial_balance: 初始资金
        """
        self.initial_balance = initial_balance
        self.cash = initial_balance
        self.positions = {}  # {symbol: quantity}
        self.trade_history = []
        self.portfolio_history = []
        
    def get_total_value(self, prices: Dict[str, float]) -> float:
        """
        计算投资组合总价值
        
        Args:
            prices: 各币种当前价格字典
            
        Returns:
            投资组合总价值
        """
        total_value = self.cash
        
        for symbol, quantity in self.positions.items():
            if symbol in prices and prices[symbol] is not None:
                total_value += quantity * prices[symbol]
        
        return total_value
    
    def get_position_value(self, symbol: str, price: float) -> float:
        """
        获取特定币种的持仓价值
        
        Args:
            symbol: 币种符号
            price: 当前价格
            
        Returns:
            持仓价值
        """
        if symbol not in self.positions:
            return 0.0
        return self.positions[symbol] * price
    
    def buy(self, symbol: str, amount: float, price: float, timestamp: datetime = None) -> bool:
        """
        买入操作
        
        Args:
            symbol: 币种符号
            amount: 买入金额
            price: 买入价格
            timestamp: 交易时间
            
        Returns:
            是否成功买入
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        if amount > self.cash:
            print(f"资金不足，无法买入 {symbol}，需要 {amount:.2f}，可用 {self.cash:.2f}")
            return False
        
        quantity = amount / price
        
        # 更新持仓
        if symbol in self.positions:
            self.positions[symbol] += quantity
        else:
            self.positions[symbol] = quantity
        
        # 更新现金
        self.cash -= amount
        
        # 记录交易
        trade = {
            'timestamp': timestamp,
            'action': 'BUY',
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'amount': amount,
            'cash_after': self.cash
        }
        self.trade_history.append(trade)
        
        print(f"买入 {symbol}: {quantity:.6f} 个，价格 {price:.2f}，金额 {amount:.2f}")
        return True
    
    def sell(self, symbol: str, amount: float, price: float, timestamp: datetime = None) -> bool:
        """
        卖出操作
        
        Args:
            symbol: 币种符号
            amount: 卖出金额
            price: 卖出价格
            timestamp: 交易时间
            
        Returns:
            是否成功卖出
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        if symbol not in self.positions:
            print(f"没有 {symbol} 持仓，无法卖出")
            return False
        
        quantity_to_sell = amount / price
        
        if quantity_to_sell > self.positions[symbol]:
            print(f"{symbol} 持仓不足，无法卖出 {quantity_to_sell:.6f}，持有 {self.positions[symbol]:.6f}")
            return False
        
        # 更新持仓
        self.positions[symbol] -= quantity_to_sell
        if self.positions[symbol] <= 1e-8:  # 避免浮点数精度问题
            del self.positions[symbol]
        
        # 更新现金
        self.cash += amount
        
        # 记录交易
        trade = {
            'timestamp': timestamp,
            'action': 'SELL',
            'symbol': symbol,
            'quantity': quantity_to_sell,
            'price': price,
            'amount': amount,
            'cash_after': self.cash
        }
        self.trade_history.append(trade)
        
        print(f"卖出 {symbol}: {quantity_to_sell:.6f} 个，价格 {price:.2f}，金额 {amount:.2f}")
        return True
    
    def sell_all(self, symbol: str, price: float, timestamp: datetime = None) -> bool:
        """
        卖出所有持仓
        
        Args:
            symbol: 币种符号
            price: 卖出价格
            timestamp: 交易时间
            
        Returns:
            是否成功卖出
        """
        if symbol not in self.positions:
            return False
        
        quantity = self.positions[symbol]
        amount = quantity * price
        
        return self.sell(symbol, amount, price, timestamp)
    
    def rebalance_to_cash(self, prices: Dict[str, float], timestamp: datetime = None):
        """
        将所有持仓转换为现金
        
        Args:
            prices: 各币种当前价格
            timestamp: 交易时间
        """
        for symbol in list(self.positions.keys()):
            if symbol in prices and prices[symbol] is not None:
                self.sell_all(symbol, prices[symbol], timestamp)
    
    def record_portfolio_state(self, timestamp: datetime, prices: Dict[str, float]):
        """
        记录投资组合状态
        
        Args:
            timestamp: 时间戳
            prices: 各币种价格
        """
        total_value = self.get_total_value(prices)
        
        state = {
            'timestamp': timestamp,
            'total_value': total_value,
            'cash': self.cash,
            'positions': self.positions.copy(),
            'return_pct': (total_value - self.initial_balance) / self.initial_balance * 100
        }
        
        self.portfolio_history.append(state)
    
    def get_performance_summary(self) -> Dict:
        """
        获取投资组合表现摘要
        
        Returns:
            表现摘要字典
        """
        if not self.portfolio_history:
            return {}
        
        df = pd.DataFrame(self.portfolio_history)
        
        final_value = df['total_value'].iloc[-1]
        total_return = (final_value - self.initial_balance) / self.initial_balance * 100
        
        # 计算最大回撤
        rolling_max = df['total_value'].expanding().max()
        drawdown = (df['total_value'] - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()
        
        # 计算夏普比率（简化版，假设无风险利率为0）
        returns = df['total_value'].pct_change().dropna()
        if len(returns) > 1:
            sharpe_ratio = returns.mean() / returns.std() * (252 ** 0.5)  # 年化
        else:
            sharpe_ratio = 0
        
        return {
            'initial_balance': self.initial_balance,
            'final_value': final_value,
            'total_return_pct': total_return,
            'max_drawdown_pct': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': len(self.trade_history),
            'trading_days': len(self.portfolio_history)
        }
    
    def get_trade_history_df(self) -> pd.DataFrame:
        """
        获取交易历史DataFrame
        
        Returns:
            交易历史DataFrame
        """
        if not self.trade_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trade_history)
    
    def get_portfolio_history_df(self) -> pd.DataFrame:
        """
        获取投资组合历史DataFrame
        
        Returns:
            投资组合历史DataFrame
        """
        if not self.portfolio_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.portfolio_history)
    
    def print_status(self, prices: Dict[str, float] = None):
        """
        打印当前状态
        
        Args:
            prices: 当前价格字典
        """
        print(f"\n=== 投资组合状态 ===")
        print(f"现金: {self.cash:.2f}")
        
        if self.positions:
            print("持仓:")
            total_position_value = 0
            for symbol, quantity in self.positions.items():
                if prices and symbol in prices and prices[symbol] is not None:
                    value = quantity * prices[symbol]
                    total_position_value += value
                    print(f"  {symbol}: {quantity:.6f} 个 (价值: {value:.2f})")
                else:
                    print(f"  {symbol}: {quantity:.6f} 个")
        
        if prices:
            total_value = self.get_total_value(prices)
            print(f"总价值: {total_value:.2f}")
            print(f"总收益率: {(total_value - self.initial_balance) / self.initial_balance * 100:.2f}%")