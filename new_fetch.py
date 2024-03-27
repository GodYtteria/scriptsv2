import ccxt
import pandas as pd
from datetime import datetime
import pytz

def fetch_data(symbol, timeframe):
    binance = ccxt.binance()  # Initialize the Binance exchange object inside the function
    ohlcv = binance.fetch_ohlcv(symbol, timeframe)
    data = []
    utc_zone = pytz.utc  # Define the UTC timezone
    for entry in ohlcv:
        timestamp = entry[0]
        dt_object = datetime.fromtimestamp(timestamp / 1000, tz=utc_zone)  # Make datetime object timezone-aware
        data.append({
            'Date': dt_object.strftime('%Y-%m-%d %H:%M:%S'),  
            'Category': 'Crypto', 
            'Symbol': symbol,
            'Open': entry[1],
            'High': entry[2],
            'Low': entry[3],
            'Close': entry[4],
            'Volume': entry[5],
        })
    return data

def read_existing_data(filepath):
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        return pd.DataFrame()

def merge_and_update_data(new_data, existing_data):
    new_df = pd.DataFrame(new_data)
    new_df['Date'] = pd.to_datetime(new_df['Date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    
    if not existing_data.empty:
        existing_data['Date'] = pd.to_datetime(existing_data['Date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        updated_df = pd.concat([existing_data, new_df]).drop_duplicates(['Date', 'Symbol', 'Category'], keep='last')
    else:
        updated_df = new_df
    
    return updated_df
#, 'ETH/USDT', 'XRP/USDT', 'LTC/USDT', 'BNB/USDT', 'ADA/USDT', 'XLM/USDT', 'DOGE/USDT', 'SOL/USDT'
#symbols = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'LTC/USDT', 'BNB/USDT', 'ADA/USDT', 'XLM/USDT', 'DOGE/USDT', 'SOL/USDT'] 
symbols = ['USDT/IDRT', 'USDT/BIDR', 'USDT/TRY', 'USDT/ARS', 'BTC/USDT', 'ETH/USDT', 'USDC/USDT', 'FDUSD/USDT', 'SOL/USDT', 'DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT', 'WIF/USDT', 'ETHFI/USDT', 'XRP/USDT', 'BNB/USDT', 'FET/USDT', 'BOME/USDT', 'SUI/USDT', 'LTC/USDT', 'NEAR/USDT', 'ICP/USDT', 'RUNE/USDT', 'AVAX/USDT', 'APT/USDT', 'FLOKI/USDT', 'T/USDT', 'FTM/USDT', 'ARB/USDT', 'POLYX/USDT'] 
#symbols = ['BTC/USDT']
timeframes = {'1h': '1h_crypto_data.csv', '12h': '1d_crypto_data.csv',  '1d': '3d_crypto_data.csv','3d': '1w_crypto_data.csv'}

while True:  # Run the script indefinitely
    for timeframe, csv_filename in timeframes.items():
        all_new_data = []
        for symbol in symbols:
            symbol_data = fetch_data(symbol, timeframe)
            all_new_data.extend(symbol_data)

        existing_data = read_existing_data(csv_filename)
        updated_data = merge_and_update_data(all_new_data, existing_data)

        updated_data.to_csv(csv_filename, index=False)
        print(f'Data for {timeframe} timeframe exported to {csv_filename}')
        print("Done")
