import vectorbt as vbt

start = "2019-01-01 UTC"  # crypto is in UTC
end = "2020-01-01 UTC"
btc_price = vbt.YFData.download("BTC-USD", start=start, end=end).get("Close")

# print(btc_price)

fast_ma = vbt.MA.run(btc_price, 10, short_name="fast")
slow_ma = vbt.MA.run(btc_price, 20, short_name="slow")

entries = fast_ma.ma_crossed_above(slow_ma)
exits = fast_ma.ma_crossed_below(slow_ma)

pf = vbt.Portfolio.from_signals(btc_price, entries, exits)

print(pf.stats())
