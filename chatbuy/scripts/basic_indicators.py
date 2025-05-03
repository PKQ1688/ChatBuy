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
    if len(data) == 0:
        raise ValueError("No data available to calculate indicators")

    # 参考 talipp 官方示例风格，逐步添加数据并获取指标
    macd = MACD(12, 26, 9)
    bb = BB(20, 2)
    rsi = RSI(14)
    adx = ADX(14, 14)

    # 初始化指标列
    data["macd"] = None
    data["signal"] = None
    data["histogram"] = None
    data["bb_upper"] = None
    data["bb_middle"] = None
    data["bb_lower"] = None
    data["rsi"] = None
    data["adx"] = None
    data["di_plus"] = None
    data["di_minus"] = None

    # 逐行添加数据
    for i, row in data.iterrows():
        ohlcv = OHLCV(
            open=row["Open"],
            high=row["High"],
            low=row["Low"],
            close=row["Close"],
            volume=row["Volume"],
        )
        macd.add(ohlcv.close)
        bb.add(ohlcv.close)
        rsi.add(ohlcv.close)
        adx.add(ohlcv)

        # MACD
        macd_val = macd[-1] if len(macd) > 0 else None
        if macd_val is not None:
            data.at[i, "macd"] = macd_val.macd
            data.at[i, "signal"] = macd_val.signal
            data.at[i, "histogram"] = macd_val.histogram

        # BB
        bb_val = bb[-1] if len(bb) > 0 else None
        if bb_val is not None:
            data.at[i, "bb_upper"] = bb_val.ub
            data.at[i, "bb_middle"] = bb_val.cb
            data.at[i, "bb_lower"] = bb_val.lb

        # RSI
        rsi_val = rsi[-1] if len(rsi) > 0 else None
        if rsi_val is not None:
            if hasattr(rsi_val, "value"):
                data.at[i, "rsi"] = rsi_val.value
            else:
                data.at[i, "rsi"] = rsi_val

        # ADX
        adx_val = adx[-1] if len(adx) > 0 else None
        if adx_val is not None:
            data.at[i, "adx"] = adx_val.adx
            data.at[i, "di_plus"] = adx_val.plus_di
            data.at[i, "di_minus"] = adx_val.minus_di

    data = data.infer_objects(copy=False)
    # 删除包含任何 NaN 的行（即有指标未计算出的行）
    data = data.dropna().reset_index(drop=True)
    return data


if __name__ == "__main__":
    data = pd.read_csv("data/BTC_USDT_1d_raw.csv")

    # 计算并添加指标
    data = add_basic_indicators(data)
    data.to_csv("data/BTC_USDT_1d_with_indicators.csv", index=False)
