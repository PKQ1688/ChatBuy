import pandas as pd
from talipp.indicators import ADX, BB, MACD, RSI
from talipp.ohlcv import OHLCV


def add_basic_indicators(data):
    """Add basic technical indicators (MACD, Bollinger Bands, RSI, ADX) to the given DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing OHLCV data with columns: 'open', 'high', 'low', 'close', 'volume'.

    Returns:
    -------
    pd.DataFrame
        The input DataFrame with additional columns for the calculated indicators.
    """
    ohlcv_list = [
        OHLCV(open=row["open"], high=row["high"], low=row["low"], close=row["close"], volume=row["volume"])
        for _, row in data.iterrows()
    ]
    macd = MACD(12, 26, 9, [x.close for x in ohlcv_list])[:]
    bb = BB(20, 2, [x.close for x in ohlcv_list])[:]
    rsi = RSI(14, [x.close for x in ohlcv_list])[:]
    adx = ADX(14, ohlcv_list)[:]

    for index, row in data.iterrows():
        if macd[index] is not None:
            data.at[index, "macd"] = macd[index].macd
            data.at[index, "signal"] = macd[index].signal
            data.at[index, "histogram"] = macd[index].histogram

        if bb[index] is not None:
            data.at[index, "bb_upper"] = bb[index].ub
            data.at[index, "bb_middle"] = bb[index].cb
            data.at[index, "bb_lower"] = bb[index].lb

        if rsi[index] is not None:
            data.at[index, "rsi"] = rsi[index].value

        if adx[index] is not None:
            data.at[index, "adx"] = adx[index].adx
            data.at[index, "di_plus"] = adx[index].di_plus
            data.at[index, "di_minus"] = adx[index].di_minus

    data = data.infer_objects(copy=False)
    data.fillna(0, inplace=True)
    return data

if __name__ == "__main__":
    data = pd.read_csv("data/BTC_USDT_1d.csv")

    # 计算并添加指标
    data = add_basic_indicators(data)
    data.to_csv("data/BTC_USDT_1d_with_indicators.csv", index=False)
