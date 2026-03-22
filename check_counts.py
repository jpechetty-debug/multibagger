import ticker_list

for bundle_name, tickers in ticker_list.BUNDLES.items():
    print(f"{bundle_name}: {len(tickers)} stocks")

# Total unique stocks across all bundles
all_tickers = set()
for tickers in ticker_list.BUNDLES.values():
    all_tickers.update(tickers)
print(f"TOTAL UNIQUE STOCKS: {len(all_tickers)}")
