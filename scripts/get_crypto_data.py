import os
import time

import ccxt
import pandas as pd

# 初始化 Binance 客户端
exchange = ccxt.binance({'rateLimit': 1200, 'enableRateLimit': True})


def fetch_historical_data(symbol, timeframe, start_date, limit=1000):
    """Fetch historical K-line data from Binance in batches.

    :param symbol: Trading pair (e.g., 'BTC/USDT')
    :param timeframe: Time period (e.g., '1m', '1h', '1d')
    :param start_date: Start time (ISO format, e.g., '2017-07-01T00:00:00Z')
    :param limit: Maximum number of entries per request (default 1000)
    :return: DataFrame containing all historical data
    """
    all_data = []
    since = exchange.parse8601(start_date)

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                break
            all_data.extend(ohlcv)
            since = ohlcv[-1][0] + 1  # 下一批次从最新时间开始
            print(f'Fetched {len(ohlcv)} rows. Current timestamp: {ohlcv[-1][0]}')
            time.sleep(exchange.rateLimit / 1000)  # 避免触发速率限制
        except Exception as e:
            print(f'Error fetching data: {e}')
            break

    # 转换为 DataFrame
    df = pd.DataFrame(
        all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df


def main(symbol='BTC', timeframe='1d', start_date='2017-07-01T00:00:00Z'):
    """Fetch and save historical cryptocurrency data from Binance.

    Args:
        symbol (str): The cryptocurrency symbol (default: 'BTC')
        timeframe (str): The time interval for data points (default: '1d')
        start_date (str): The start date in ISO format (default: '2017-07-01T00:00:00Z')
    """
    # 获取 交易对历史数据
    symbol_pair = f'{symbol}/USDT'
    timeframe = timeframe  # 每天的数据
    start_date = start_date  # Binance 上线日期

    df = fetch_historical_data(symbol_pair, timeframe, start_date)

    # 创建 data 目录（如果不存在）
    if not os.path.exists('data'):
        os.makedirs('data')

    # 保存为 CSV 文件
    df.to_csv(f'data/{symbol}_USDT_{timeframe}.csv', index=False)
    print(f'history data is store in {symbol}_USDT_{timeframe}.csv')


if __name__ == '__main__':
    main()
