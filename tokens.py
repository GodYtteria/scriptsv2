import ccxt

# Initialize the Binance exchange
binance = ccxt.binance()

# Fetch all markets
markets = binance.load_markets()

# Fetch tickers for all markets
tickers = binance.fetch_tickers()

# Filter and sort by 24h volume in descending order, considering only 'USDT' pairs
sorted_tickers = sorted(
    [(symbol, ticker) for symbol, ticker in tickers.items() if 'USDT' in symbol],
    key=lambda x: x[1]['quoteVolume'],
    reverse=True
)

# Extract the top 30 symbols
top_30_symbols = [ticker[0] for ticker in sorted_tickers[:30]]

print(top_30_symbols)
