import ccxt
import pandas as pd
from datetime import datetime
import pytz
import time
import os
import asyncio

# Capture the start time
start_time = time.time()

def fetch_data(symbol, timeframe, binance):
    # Fetch OHLCV data for a given symbol and timeframe from the exchange
    print(f"Fetching OHLCV data for {symbol} - {timeframe}...")
    ohlcv = binance.fetch_ohlcv(symbol, timeframe)
    data = []
    utc_zone = pytz.utc
    # Convert timestamp to datetime and store data in a list of dictionaries
    for entry in ohlcv:
        timestamp = entry[0]
        dt_object = datetime.fromtimestamp(timestamp / 1000, tz=utc_zone)
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
        # Read existing data from CSV file
        print(f"Reading existing data from {filepath}...")
        return pd.read_csv(filepath)
    except FileNotFoundError:
        # If file not found, return an empty DataFrame
        print("Existing data file not found.")
        return pd.DataFrame()

def downsample_data(df, timeframe):
    if timeframe == '1h':
        return df.iloc[::3]  # Take every 3rd row for 1-hour timeframe
    elif timeframe == '12h':
        return df.iloc[::2]  # Take every 2nd row for 12-hour timeframe
    elif timeframe in ['1d', '3d']:
        return df.iloc[::2]  # Take every 3rd row for 1-day and 3-day timeframes
    return df  # Return the original DataFrame for other timeframes

def merge_and_update_data(new_data, existing_data, timeframe):
    # Merge new data with existing data and remove duplicates
    if not new_data:
        return existing_data  # Return existing data if new_data is empty
    
    new_df = pd.DataFrame(new_data)
    new_df['Date'] = pd.to_datetime(new_df['Date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    if not existing_data.empty:
        existing_data['Date'] = pd.to_datetime(existing_data['Date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        updated_df = pd.concat([existing_data, new_df]).drop_duplicates(['Date', 'Symbol', 'Category'], keep='last')
    else:
        updated_df = new_df

    # Downsample data based on timeframe
    updated_df = downsample_data(updated_df, timeframe)

    return updated_df

# List of symbols to fetch data for
symbols = ['USDT/IDRT', 'USDT/BIDR', 'USDT/TRY', 'USDT/ARS', 'BTC/USDT', 'ETH/USDT', 'USDC/USDT', 'FDUSD/USDT',
           'SOL/USDT', 'DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT', 'WIF/USDT', 'ETHFI/USDT', 'XRP/USDT', 'BNB/USDT',
           'FET/USDT', 'BOME/USDT', 'SUI/USDT', 'LTC/USDT', 'NEAR/USDT', 'ICP/USDT', 'RUNE/USDT', 'AVAX/USDT',
           'APT/USDT', 'FLOKI/USDT', 'T/USDT', 'FTM/USDT', 'ARB/USDT', 'POLYX/USDT']

# Dictionary of timeframes and corresponding CSV filenames
timeframes = {'1h': '1h_crypto_data.csv', '12h': '1d_crypto_data.csv', '1d': '3d_crypto_data.csv','3d': '1w_crypto_data.csv'}

async def main():
    binance = ccxt.binance()
    for timeframe, csv_filename in timeframes.items():
        print(f'Fetching data for {timeframe} timeframe...')
        all_new_data = []
        # Fetch data for each symbol and append to all_new_data
        for symbol in symbols:
            symbol_data = fetch_data(symbol, timeframe, binance)
            all_new_data.extend(symbol_data)
            print(f'{len(symbol_data)} entries fetched for {symbol}.')
        existing_data = read_existing_data(csv_filename)
        updated_data = merge_and_update_data(all_new_data, existing_data, timeframe)
        if not updated_data.equals(existing_data):  # Check if data has been updated
            updated_data.to_csv(csv_filename, index=False)
            print(f'Data for {timeframe} timeframe exported to {csv_filename}')
        else:
            print('No new data found. Skipping export.')

if __name__ == "__main__":
    asyncio.run(main())

    # Calculate and print the duration
    end_time = time.time()
    duration = end_time - start_time
    minutes = duration // 60
    seconds = duration % 60
    print(f"Finished in {int(minutes)} minutes and {int(seconds)} seconds.")
