"""
数据管理器，用于获取和管理加密货币数据
"""

import os
import pandas as pd
import ccxt
from typing import Dict, Optional
from datetime import datetime, timedelta
import time


class CryptoDataManager:
    """加密货币数据管理器"""
    
    def __init__(self, data_dir: str = "data"):
        """
        初始化数据管理器
        
        Args:
            data_dir: 数据存储目录
        """
        self.data_dir = data_dir
        self.exchange = ccxt.binance({"rateLimit": 1200, "enableRateLimit": True})
        
        # 确保数据目录存在
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
    def get_historical_data(self, symbol: str, timeframe: str = "1d", 
                          days: int = 365) -> pd.DataFrame:
        """
        获取历史数据
        
        Args:
            symbol: 交易对符号 (如 'BTC', 'ETH')
            timeframe: 时间周期 (如 '1d', '4h', '1h')
            days: 获取多少天的数据
            
        Returns:
            包含OHLCV数据的DataFrame
        """
        symbol_pair = f"{symbol}/USDT"
        filename = f"{symbol}_USDT_{timeframe}_{days}d.csv"
        filepath = os.path.join(self.data_dir, filename)
        
        # 检查是否有缓存数据
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            # 检查数据是否需要更新（如果最新数据超过1天）
            if datetime.now() - df.index[-1] < timedelta(days=1):
                return df
        
        # 获取新数据
        start_date = datetime.now() - timedelta(days=days)
        start_date_str = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        try:
            df = self._fetch_historical_data(symbol_pair, timeframe, start_date_str)
            # 保存到缓存
            df.to_csv(filepath)
            return df
        except Exception as e:
            print(f"获取 {symbol} 数据失败: {e}")
            # 如果有缓存数据，返回缓存数据
            if os.path.exists(filepath):
                return pd.read_csv(filepath, index_col=0, parse_dates=True)
            raise
    
    def _fetch_historical_data(self, symbol: str, timeframe: str, 
                              start_date: str, limit: int = 1000) -> pd.DataFrame:
        """
        从交易所获取历史数据
        """
        all_data = []
        since = self.exchange.parse8601(start_date)
        
        while True:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
                
                if not ohlcv:
                    break
                
                all_data.extend(ohlcv)
                
                # 检查是否到达当前时间
                last_timestamp = ohlcv[-1][0]
                current_timestamp = self.exchange.milliseconds()
                since = last_timestamp + 1
                
                if last_timestamp >= current_timestamp - self.exchange.parse_timeframe(timeframe) * 1000:
                    break
                
                # 遵守速率限制
                time.sleep(self.exchange.rateLimit / 1000)
                
            except Exception as e:
                print(f"获取数据时出错: {e}")
                time.sleep(5)
                continue
        
        if not all_data:
            raise ValueError(f"无法获取 {symbol} 的数据")
        
        # 转换为DataFrame
        df = pd.DataFrame(
            all_data, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        
        # 标准化列名
        df.rename(columns={
            "open": "Open",
            "high": "High", 
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        }, inplace=True)
        
        return df
    
    def get_current_price(self, symbol: str) -> float:
        """
        获取当前价格
        
        Args:
            symbol: 交易对符号 (如 'BTC', 'ETH')
            
        Returns:
            当前价格
        """
        try:
            ticker = self.exchange.fetch_ticker(f"{symbol}/USDT")
            return ticker['last']
        except Exception as e:
            print(f"获取 {symbol} 当前价格失败: {e}")
            return None
    
    def get_multiple_data(self, symbols: list, timeframe: str = "1d", 
                         days: int = 365) -> Dict[str, pd.DataFrame]:
        """
        获取多个币种的数据
        
        Args:
            symbols: 币种列表
            timeframe: 时间周期
            days: 天数
            
        Returns:
            包含各币种数据的字典
        """
        data = {}
        for symbol in symbols:
            try:
                data[symbol] = self.get_historical_data(symbol, timeframe, days)
                print(f"成功获取 {symbol} 数据，共 {len(data[symbol])} 条记录")
            except Exception as e:
                print(f"获取 {symbol} 数据失败: {e}")
        
        return data