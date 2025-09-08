"""
BTC和ETH轮动策略实现
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from .data_manager import CryptoDataManager
from .portfolio import Portfolio


class BTCETHRotationStrategy:
    """BTC和ETH轮动策略"""
    
    def __init__(self, 
                 initial_balance: float = 10000.0,
                 lookback_period: int = 20,
                 momentum_period: int = 10,
                 rsi_period: int = 14,
                 rebalance_frequency: str = "weekly"):
        """
        初始化轮动策略
        
        Args:
            initial_balance: 初始资金
            lookback_period: 回望期（用于计算技术指标）
            momentum_period: 动量计算期
            rsi_period: RSI计算期
            rebalance_frequency: 再平衡频率 ("daily", "weekly", "monthly")
        """
        self.initial_balance = initial_balance
        self.lookback_period = lookback_period
        self.momentum_period = momentum_period
        self.rsi_period = rsi_period
        self.rebalance_frequency = rebalance_frequency
        
        self.data_manager = CryptoDataManager()
        self.portfolio = Portfolio(initial_balance)
        
        self.symbols = ["BTC", "ETH"]
        self.data = {}
        self.signals = {}
        
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        
        Args:
            df: OHLCV数据
            
        Returns:
            包含技术指标的DataFrame
        """
        df = df.copy()
        
        # 计算移动平均线
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # 计算MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # 计算RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 计算布林带
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # 计算动量指标
        df['Momentum'] = df['Close'] / df['Close'].shift(self.momentum_period) - 1
        df['Price_Change'] = df['Close'].pct_change()
        
        # 计算波动率
        df['Volatility'] = df['Price_Change'].rolling(window=20).std() * np.sqrt(252)
        
        # 计算相对强度（与另一个资产比较时使用）
        df['Returns'] = df['Close'].pct_change()
        df['Cumulative_Returns'] = (1 + df['Returns']).cumprod()
        
        return df
    
    def calculate_rotation_signals(self, btc_data: pd.DataFrame, eth_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算轮动信号
        
        Args:
            btc_data: BTC数据
            eth_data: ETH数据
            
        Returns:
            包含各种信号的字典
        """
        signals = {}
        
        # 确保两个数据集有相同的日期索引
        common_dates = btc_data.index.intersection(eth_data.index)
        btc_aligned = btc_data.loc[common_dates]
        eth_aligned = eth_data.loc[common_dates]
        
        # 1. 动量轮动信号
        btc_momentum = btc_aligned['Momentum']
        eth_momentum = eth_aligned['Momentum']
        signals['momentum_signal'] = np.where(btc_momentum > eth_momentum, 'BTC', 'ETH')
        
        # 2. RSI轮动信号（选择RSI更低的，即更超卖的）
        btc_rsi = btc_aligned['RSI']
        eth_rsi = eth_aligned['RSI']
        signals['rsi_signal'] = np.where(btc_rsi < eth_rsi, 'BTC', 'ETH')
        
        # 3. 趋势跟随信号（MACD）
        btc_macd_signal = btc_aligned['MACD'] > btc_aligned['MACD_Signal']
        eth_macd_signal = eth_aligned['MACD'] > eth_aligned['MACD_Signal']
        
        # 选择MACD信号更强的
        btc_macd_strength = btc_aligned['MACD'] - btc_aligned['MACD_Signal']
        eth_macd_strength = eth_aligned['MACD'] - eth_aligned['MACD_Signal']
        signals['macd_signal'] = np.where(btc_macd_strength > eth_macd_strength, 'BTC', 'ETH')
        
        # 4. 相对强度信号
        btc_returns = btc_aligned['Returns']
        eth_returns = eth_aligned['Returns']
        relative_strength = btc_returns.rolling(window=20).mean() - eth_returns.rolling(window=20).mean()
        signals['relative_strength_signal'] = np.where(relative_strength > 0, 'BTC', 'ETH')
        
        # 5. 波动率调整信号（选择风险调整后收益更高的）
        btc_risk_adj_return = btc_aligned['Returns'].rolling(window=20).mean() / btc_aligned['Volatility']
        eth_risk_adj_return = eth_aligned['Returns'].rolling(window=20).mean() / eth_aligned['Volatility']
        signals['risk_adj_signal'] = np.where(btc_risk_adj_return > eth_risk_adj_return, 'BTC', 'ETH')
        
        # 6. 综合信号（投票机制）
        signal_df = pd.DataFrame(signals, index=common_dates)
        
        # 计算每个资产的得票数
        btc_votes = (signal_df == 'BTC').sum(axis=1)
        eth_votes = (signal_df == 'ETH').sum(axis=1)
        
        signals['composite_signal'] = np.where(btc_votes > eth_votes, 'BTC', 'ETH')
        
        # 转换为pandas Series
        for key, value in signals.items():
            if not isinstance(value, pd.Series):
                signals[key] = pd.Series(value, index=common_dates)
        
        return signals
    
    def should_rebalance(self, current_date: datetime, last_rebalance_date: datetime) -> bool:
        """
        判断是否需要再平衡
        
        Args:
            current_date: 当前日期
            last_rebalance_date: 上次再平衡日期
            
        Returns:
            是否需要再平衡
        """
        if self.rebalance_frequency == "daily":
            return True
        elif self.rebalance_frequency == "weekly":
            return (current_date - last_rebalance_date).days >= 7
        elif self.rebalance_frequency == "monthly":
            return (current_date - last_rebalance_date).days >= 30
        else:
            return False
    
    def execute_rotation(self, target_symbol: str, current_prices: Dict[str, float], 
                        timestamp: datetime) -> bool:
        """
        执行轮动操作
        
        Args:
            target_symbol: 目标持仓币种
            current_prices: 当前价格
            timestamp: 时间戳
            
        Returns:
            是否执行了交易
        """
        # 获取当前持仓
        current_positions = list(self.portfolio.positions.keys())
        
        # 如果已经持有目标币种，不需要操作
        if len(current_positions) == 1 and current_positions[0] == target_symbol:
            return False
        
        # 清空所有持仓
        self.portfolio.rebalance_to_cash(current_prices, timestamp)
        
        # 买入目标币种
        if target_symbol in current_prices and current_prices[target_symbol] is not None:
            # 使用95%的资金买入，保留5%作为缓冲
            buy_amount = self.portfolio.cash * 0.95
            success = self.portfolio.buy(target_symbol, buy_amount, 
                                       current_prices[target_symbol], timestamp)
            return success
        
        return False
    
    def backtest(self, start_date: str = None, end_date: str = None, 
                 days: int = 365) -> Dict:
        """
        回测策略
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            days: 回测天数（如果未指定日期）
            
        Returns:
            回测结果
        """
        print("开始获取数据...")
        
        # 获取数据
        self.data = self.data_manager.get_multiple_data(self.symbols, "1d", days)
        
        if len(self.data) != 2:
            raise ValueError("无法获取完整的BTC和ETH数据")
        
        # 计算技术指标
        print("计算技术指标...")
        for symbol in self.symbols:
            self.data[symbol] = self.calculate_technical_indicators(self.data[symbol])
        
        # 计算轮动信号
        print("计算轮动信号...")
        self.signals = self.calculate_rotation_signals(self.data["BTC"], self.data["ETH"])
        
        # 获取共同的交易日期
        common_dates = self.data["BTC"].index.intersection(self.data["ETH"].index)
        common_dates = common_dates.sort_values()
        
        # 过滤掉前面没有足够数据计算指标的日期
        start_idx = max(self.lookback_period, self.rsi_period, 50)  # 确保有足够数据
        trading_dates = common_dates[start_idx:]
        
        print(f"开始回测，交易日期从 {trading_dates[0]} 到 {trading_dates[-1]}，共 {len(trading_dates)} 天")
        
        last_rebalance_date = trading_dates[0] - timedelta(days=30)  # 确保第一天会触发再平衡
        current_target = None
        
        for i, date in enumerate(trading_dates):
            # 获取当前价格
            current_prices = {
                "BTC": self.data["BTC"].loc[date, "Close"],
                "ETH": self.data["ETH"].loc[date, "Close"]
            }
            
            # 获取当前信号
            target_symbol = self.signals['composite_signal'].loc[date]
            
            # 判断是否需要再平衡
            if (self.should_rebalance(date, last_rebalance_date) or 
                target_symbol != current_target):
                
                # 执行轮动
                if self.execute_rotation(target_symbol, current_prices, date):
                    last_rebalance_date = date
                    current_target = target_symbol
                    print(f"{date.strftime('%Y-%m-%d')}: 轮动到 {target_symbol}")
            
            # 记录投资组合状态
            self.portfolio.record_portfolio_state(date, current_prices)
            
            # 每100天打印一次进度
            if (i + 1) % 100 == 0:
                total_value = self.portfolio.get_total_value(current_prices)
                print(f"进度: {i+1}/{len(trading_dates)}, 当前价值: {total_value:.2f}")
        
        # 计算基准表现（买入并持有BTC和ETH各50%）
        benchmark_performance = self.calculate_benchmark_performance(trading_dates)
        
        # 获取策略表现
        strategy_performance = self.portfolio.get_performance_summary()
        
        print("\n=== 回测完成 ===")
        print(f"策略总收益率: {strategy_performance['total_return_pct']:.2f}%")
        print(f"基准总收益率: {benchmark_performance['total_return_pct']:.2f}%")
        print(f"超额收益: {strategy_performance['total_return_pct'] - benchmark_performance['total_return_pct']:.2f}%")
        
        return {
            'strategy_performance': strategy_performance,
            'benchmark_performance': benchmark_performance,
            'portfolio_history': self.portfolio.get_portfolio_history_df(),
            'trade_history': self.portfolio.get_trade_history_df(),
            'signals': pd.DataFrame(self.signals)
        }
    
    def calculate_benchmark_performance(self, trading_dates: pd.DatetimeIndex) -> Dict:
        """
        计算基准表现（50% BTC + 50% ETH买入并持有）
        
        Args:
            trading_dates: 交易日期
            
        Returns:
            基准表现字典
        """
        start_date = trading_dates[0]
        end_date = trading_dates[-1]
        
        start_btc_price = self.data["BTC"].loc[start_date, "Close"]
        start_eth_price = self.data["ETH"].loc[start_date, "Close"]
        end_btc_price = self.data["BTC"].loc[end_date, "Close"]
        end_eth_price = self.data["ETH"].loc[end_date, "Close"]
        
        # 50%资金买BTC，50%买ETH
        btc_quantity = (self.initial_balance * 0.5) / start_btc_price
        eth_quantity = (self.initial_balance * 0.5) / start_eth_price
        
        final_value = btc_quantity * end_btc_price + eth_quantity * end_eth_price
        total_return = (final_value - self.initial_balance) / self.initial_balance * 100
        
        return {
            'initial_balance': self.initial_balance,
            'final_value': final_value,
            'total_return_pct': total_return
        }
    
    def get_signal_analysis(self) -> pd.DataFrame:
        """
        获取信号分析
        
        Returns:
            信号分析DataFrame
        """
        if not self.signals:
            return pd.DataFrame()
        
        signal_df = pd.DataFrame(self.signals)
        
        # 计算各信号的准确性和一致性
        analysis = {}
        
        for signal_name in signal_df.columns:
            if signal_name != 'composite_signal':
                signal_series = signal_df[signal_name]
                
                # 计算信号变化频率
                changes = (signal_series != signal_series.shift(1)).sum()
                
                # 计算BTC和ETH的选择比例
                btc_ratio = (signal_series == 'BTC').mean()
                eth_ratio = (signal_series == 'ETH').mean()
                
                analysis[signal_name] = {
                    'signal_changes': changes,
                    'btc_ratio': btc_ratio,
                    'eth_ratio': eth_ratio
                }
        
        return pd.DataFrame(analysis).T
    
    def plot_results(self, save_path: str = None):
        """
        绘制回测结果图表
        
        Args:
            save_path: 保存路径
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            portfolio_df = self.portfolio.get_portfolio_history_df()
            if portfolio_df.empty:
                print("没有投资组合历史数据可绘制")
                return
            
            fig, axes = plt.subplots(3, 1, figsize=(15, 12))
            
            # 1. 投资组合价值变化
            axes[0].plot(portfolio_df['timestamp'], portfolio_df['total_value'], 
                        label='策略价值', linewidth=2)
            axes[0].axhline(y=self.initial_balance, color='r', linestyle='--', 
                           label='初始资金')
            axes[0].set_title('投资组合价值变化')
            axes[0].set_ylabel('价值 (USDT)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # 2. 收益率变化
            axes[1].plot(portfolio_df['timestamp'], portfolio_df['return_pct'], 
                        label='累计收益率', color='green', linewidth=2)
            axes[1].axhline(y=0, color='r', linestyle='--')
            axes[1].set_title('累计收益率变化')
            axes[1].set_ylabel('收益率 (%)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # 3. BTC和ETH价格对比
            common_dates = self.data["BTC"].index.intersection(self.data["ETH"].index)
            btc_normalized = self.data["BTC"].loc[common_dates, "Close"] / self.data["BTC"].loc[common_dates[0], "Close"]
            eth_normalized = self.data["ETH"].loc[common_dates, "Close"] / self.data["ETH"].loc[common_dates[0], "Close"]
            
            axes[2].plot(common_dates, btc_normalized, label='BTC (标准化)', color='orange')
            axes[2].plot(common_dates, eth_normalized, label='ETH (标准化)', color='blue')
            axes[2].set_title('BTC vs ETH 价格走势 (标准化)')
            axes[2].set_ylabel('标准化价格')
            axes[2].set_xlabel('日期')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            # 格式化x轴日期
            for ax in axes:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"图表已保存到: {save_path}")
            
            plt.show()
            
        except ImportError:
            print("需要安装matplotlib来绘制图表: pip install matplotlib")
        except Exception as e:
            print(f"绘制图表时出错: {e}")