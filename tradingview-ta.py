import tradingview_ta as ta
import datetime
import pytz

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

    # Fetch technical analysis data using tradingview_ta
    ta_data = ta.get_multiple_analysis(symbol, interval=timeframe, result_output='dict')
    # Merge OHLCV data with technical analysis data
    for entry in data:
        symbol_ta = ta_data.get(entry['Symbol'], {})
        entry.update(symbol_ta)

    return data
