import pandas as pd
import talib
import numpy as np
import time
import os
import asyncio

start_time = time.time()

class TA:
    def __init__(self):
        self.MAfast, self.MAslow, self.MACDfast, self.MACDslow, self.MACDsignal, self.ATRperiod, self.ATRfactor, self.ADXperiod, self.ADXshift, self.DIperiod, self.Engulfperiod, self.RSIperiod, self.RSIupper, self.RSIlower, self.DemarkLookback = 10, 20, 12, 26, 9, 6, 2, 14, 14, 14, 10, 14, 70, 30, 10

    def MA(self, close):
        df = talib.MA(close, self.MAfast) - talib.MA(close, self.MAslow)
        df[df > 0], df[df < 0] = 1, -1
        return df
    
    def MACD_components(self, close):
        return talib.MACD(close, self.MACDfast, self.MACDslow, self.MACDsignal)

    def MACD(self, close):
        macd, macdsignal, macdhist = self.MACD_components(close)
        score = pd.Series(0.0, index=close.index)
        for i in range(1, len(close)):
            if macd[i] > macdsignal[i] and macd[i - 1] <= macdsignal[i - 1]: score[i] = 1
            elif macd[i] < macdsignal[i] and macd[i - 1] >= macdsignal[i - 1]: score[i] = -1
            if macdhist[i] > macdhist[i - 1] and macdhist[i] > 0: score[i] += 0.5
            elif macdhist[i] < macdhist[i - 1] and macdhist[i] < 0: score[i] -= 0.5
        return score
        
    def LL(self, high, low, close):
        score = pd.Series(0, index=close.index)
        score[(close > close.shift(1)) & (close > (high + low) / 2)] = 1
        score[(close < close.shift(1)) & (close < (high + low) / 2)] = -1
        return score

    def Trender(self, h, l, c):
        atr = self.ATRfactor * talib.ATR(h, l, c, self.ATRperiod)
        up, down, st, trend = 0.5 * (h + l) + atr, 0.5 * (h + l) - atr, 0.5 * (h + l) - atr, pd.Series(1, index=c.index)
        for i in range(1, c.shape[0]):
            if c.iloc[i] > st.iloc[i - 1] and trend.iloc[i - 1] < 0: trend.iloc[i], st.iloc[i] = 1, down.iloc[i]
            elif c.iloc[i] < st.iloc[i - 1] and trend.iloc[i - 1] > 0: trend.iloc[i], st.iloc[i] = -1, up.iloc[i]
            elif trend.iloc[i - 1] > 0: trend.iloc[i], st.iloc[i] = 1, max(down.iloc[i], st.iloc[i - 1])
            else: trend.iloc[i], st.iloc[i] = -1, min(up.iloc[i], st.iloc[i - 1])
        return trend

    def ADX(self, h, l, c):
        adx = talib.ADX(h, l, c, self.ADXperiod)
        adx = (adx - adx.shift(self.ADXshift)) / 2
        adx[adx > 0], adx[adx < 0] = 1, -1
        return adx

    def DMI(self, h, l ,c):
        pdi = talib.PLUS_DI(h, l, c, self.DIperiod)
        mdi = talib.MINUS_DI(h, l, c, self.DIperiod)
        dmi = pdi - mdi
        dmi[dmi > 0], dmi[dmi < 0] = 1, -1
        return dmi

    def Engulf_original(self, o, h, l, c):
        black, white, engulf = c < o, c > o, (h > h.shift(1)) & (l < l.shift(1))
        bullEngulf, bearEngulf = (white & black.shift(1) & engulf).astype(int), (black & white.shift(1) & engulf).astype(int)
        sumBull, sumBear = bullEngulf.rolling(self.Engulfperiod).sum(), bearEngulf.rolling(self.Engulfperiod).sum()
        netEngulf = 0.5 * (sumBull - sumBear)
        netEngulf[netEngulf >= 0.5], netEngulf[netEngulf <= -0.5] = 1, -1
        return netEngulf

    def Engulf(self, data, o, h, l, c):
        df, black, white, engulf = data[['OPEN', 'HIGH', 'LOW', 'CLOSE']].copy(), c < o, c > o, (h > h.shift(1)) & (l < l.shift(1)) 
        bullEngulf, bearEngulf = (white & black.shift(1) & engulf).astype(int), (black & white.shift(1) & engulf).astype(int)
        df['bullEngulf'], df['bullEngulfScore'], df['bearEngulf'], df['bearEngulfScore'], df['engulf_score'] = bullEngulf, 0, bearEngulf, 0, 0
        for i in range(len(df)):
            if df['bullEngulf'][i] == 1: 
                bull_high, bull_low = df['HIGH'][i], df['LOW'][i]
            if df['bearEngulf'][i] == 1: 
                bear_high, bear_low = df['HIGH'][i], df['LOW'][i]
            for j in range(i + 1, min(i + 11, len(df))):
                if df['bullEngulf'][i] == 1:
                    if df['CLOSE'][j] < bull_high and df['CLOSE'][j] > bull_low: 
                        df.loc[df.index[j], 'bullEngulfScore'] = -1
                    if df['CLOSE'][j] > bull_high and df['CLOSE'][j] < df['CLOSE'][j - 1]: 
                        df.loc[df.index[j], 'bullEngulfScore'] = -2
                    if df['bullEngulfScore'][j - 1] == -2 and df['bullEngulfScore'][j] != -2: 
                        df.loc[df.index[j], 'bullEngulfScore'] = 0
                        break
                    if df['CLOSE'][j] < bull_low: 
                        df.loc[df.index[j], 'bullEngulfScore'] = 0
                        break
                if df['bearEngulf'][i] == 1:
                    if df['CLOSE'][j] < bear_high and df['CLOSE'][j] > bear_low: 
                        df.loc[df.index[j], 'bearEngulfScore'] = 1
                    if df['CLOSE'][j] < bear_low and df['CLOSE'][j] < df['CLOSE'][j - 1]: 
                        df.loc[df.index[j], 'bearEngulfScore'] = 2
                    if df['bearEngulfScore'][j - 1] == 2 and df['bearEngulfScore'][j] != 2: 
                        df.loc[df.index[j], 'bearEngulfScore'] = 0
                        break
                    if df['CLOSE'][j] > bear_high: 
                        df.loc[df.index[j], 'bearEngulfScore'] = 0
                        break
        df['engulf_score'] = df['bullEngulfScore'] + df['bearEngulfScore']
        return df['engulf_score']
    
    def RSI(self, close):
        rsi, rsi_ma = talib.RSI(close, self.RSIperiod), talib.MA(talib.RSI(close, self.RSIperiod), timeperiod=5)
        rsi_score = pd.Series(0, index=rsi.index)
        for i in range(1, len(rsi)):
            if rsi.iloc[i] > self.RSIupper or (rsi.iloc[i] > rsi_ma.iloc[i] and rsi.iloc[i - 1] <= rsi_ma.iloc[i - 1]): rsi_score.iloc[i] = 1
            elif rsi.iloc[i] < self.RSIlower or (rsi.iloc[i] < rsi_ma.iloc[i] and rsi.iloc[i - 1] >= rsi_ma.iloc[i - 1]): rsi_score.iloc[i] = -1
            if rsi.iloc[i] > 70 and rsi.iloc[i] < rsi_ma.iloc[i] and rsi.iloc[i - 1] > rsi_ma.iloc[i - 1]: rsi_score.iloc[i] = 2
            elif rsi.iloc[i] < 30 and rsi.iloc[i] > rsi_ma.iloc[i] and rsi.iloc[i - 1] < rsi_ma.iloc[i - 1]: rsi_score.iloc[i] = -2
        return rsi_score
    
    def Demark(self, h, l, close):
        s = close - close.shift(4)

        setup = pd.Series(0, index=close.index)
        count = 0

        for i in range(1, close.shape[0]):
            if s.iloc[i] < 0 and s.iloc[i - 1] < 0:
                if count < 0:
                    count -= 1
                else:
                    count = -1
            elif s.iloc[i] > 0 and s.iloc[i - 1] > 0:
                if count > 0:
                    count += 1
                else:
                    count = 1
            else:
                count = 0
            setup.iloc[i] = count

        countdown = pd.Series(0, index=close.index)
        count = 0

        for i in range(1, close.shape[0]):
            if setup.iloc[i - 1] == 9:
                count = 1
            if setup.iloc[i - 1] == -9:
                count = -1

            if setup.iloc[i] > 0 and count > 0 and close.iloc[i] > h.iloc[i - 2]:
                count += 1

                if setup.iloc[i] < 0:
                    count = 0

            if setup.iloc[i] < 0 and count < 0 and close.iloc[i] < l.iloc[i - 2]:
                count -= 1

                if setup.iloc[i] > 0:
                    count = 0

            countdown.iloc[i] = count

        demark_score = pd.Series(0, index=close.index)
        up = np.nan
        down = np.nan
        up_line = pd.Series(np.nan, index=close.index)
        down_line = pd.Series(np.nan, index=close.index)

        for i in range(1, close.shape[0]):
            if setup.iloc[i] == 9 or countdown.iloc[i] == 13:
                demark_score[i] = 1
            if setup.iloc[i] == -9 or countdown.iloc[i] == -13:
                demark_score[i] = -1

            if setup.iloc[i] == 9:
                up = close[i]
                up_line[i] = up
            else:
                up_line[i] = up

            if setup.iloc[i] == -9:
                down = close[i]
                down_line[i] = down
            else:
                down_line[i] = down

            if setup.iloc[i] > 9 or countdown.iloc[i] > 13:
                if close[i] < up_line[i]:
                    demark_score[i] = 2
            if setup.iloc[i] < -9 or countdown.iloc[i] < -13:
                if close[i] > down_line[i]:
                    demark_score[i] = -2
        return demark_score

# In the process_data function, adjust the DataFrame based on the timeframe
def process_data(timeframe, input_file, output_file, merged_data=None):
    print(f"Processing data for {timeframe} timeframe...")
    
    if not os.path.exists(input_file):
        print(f"No input file found for {timeframe}, skipping...")
        return
    
    try:
        # Attempt to read the CSV file
        df = pd.read_csv(input_file)
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV file for {timeframe}: {e}")
        print("Attempting to fix CSV file...")
        try:
            # Read the CSV file while ignoring lines with too many fields
            df = pd.read_csv(input_file, error_bad_lines=False)
            print("CSV file fixed successfully.")
        except Exception as e:
            print(f"Error fixing CSV file for {timeframe}: {e}")
            return
    
    ta, df = TA(), pd.read_csv(input_file)
    df.rename(columns={'Open': 'OPEN', 'High': 'HIGH', 'Low': 'LOW', 'Close': 'CLOSE'}, inplace=True)
    df['OPEN'], df['HIGH'], df['LOW'], df['CLOSE'] = df['OPEN'].astype(float), df['HIGH'].astype(float), df['LOW'].astype(float), df['CLOSE'].astype(float)
    df['MA_Score'], df['MACD_Score'], df['LL'], df['Trender'], df['DMI'], df['Engulf'], df['RSI_Score'], df['Demark'] = ta.MA(df['CLOSE']), ta.MACD(df['CLOSE']), ta.LL(df['HIGH'], df['LOW'], df['CLOSE']), ta.Trender(df['HIGH'], df['LOW'], df['CLOSE']), ta.DMI(df['HIGH'], df['LOW'], df['CLOSE']).fillna(0), ta.Engulf(df, df['OPEN'], df['HIGH'], df['LOW'], df['CLOSE']), ta.RSI(df['CLOSE']), ta.Demark(df['HIGH'], df['LOW'], df['CLOSE'])
    df['Scores'] = df[['MA_Score', 'MACD_Score', 'LL', 'Trender', 'DMI', 'Engulf']].sum(axis=1)
    df['Extreme Scores'] = df[['RSI_Score', 'LL', 'Engulf', 'Demark']].sum(axis=1)
    #df[['MA_Score','MACD_Score','Trender', 'DMI','RSI_Score', 'LL', 'Engulf', 'Demark']].sum(axis=1)
    df['Signal'], df['Extreme Signal'] = df['Scores'].apply(lambda x: assign_signal(x)), df['Extreme Scores'].apply(lambda x: assign_extreme_signal(x))
    df.rename(columns={'CLOSE': 'PRICE'}, inplace=True)

    # Sort the DataFrame by 'Date' column in descending order
    df.sort_values(by='Date', ascending=False, inplace=True)

    # Check if the output file already exists
    if os.path.exists(output_file):
        # Append new data to the existing DataFrame
        merged_data = pd.concat([merged_data, df], ignore_index=True)
    else:
        # No existing output file found, set merged_data to the processed DataFrame
        merged_data = df
    
    # Save the merged DataFrame to the output file
    merged_data.to_csv(output_file, index=False)
    
    print(f"Processed data for {timeframe} saved to {output_file}")
    return merged_data

def assign_signal(score):
    if score > 6:
        return 'Strong Buy'
    elif score == 6:
        return 'Strong Buy'
    elif score >= 5.5:
        return 'Strong Buy'
    elif score >= 4.5:
        return 'Buy'
    elif score >= 3.5:
        return 'Neutral'
    elif score >= 2.5:
        return 'Neutral'
    elif score >= 1.5:
        return 'Neutral' if score % 1 == 0.5 else 'Neutral'  # Rounding off positive 0.5 scores
    elif score >= 0.5:
        return 'Neutral'
    elif score >= -0.5:
        return 'Neutral'
    elif score >= -1.5:
        return 'Neutral'
    elif score >= -2.5:
        return 'Neutral'
    elif score >= -3.5:
        return 'Neutral'
    elif score >= -4.5:
        return 'Sell'
    elif score >= -5.5:
        return 'Neutral'
    elif score >= -6:
        return 'Strong Sell'
    elif score < -6:
        return 'Strong Sell'
    else:
        return 'Neutral'  # Return 'Neutral' for any unexpected score

def assign_extreme_signal(score):
    global last_extreme_signal
    if score >= 2.5:
        last_extreme_signal = 'Overbought'
        return 'Overbought'
    elif score <= -2:
        last_extreme_signal = 'Oversold'
        return 'Oversold'
    else:
        return 'Neutral'


# Initialize the last extreme signal
last_extreme_signal = 'Neutral'

def read_existing_data(filepath):
    try:
        print(f"Reading existing data from {filepath}...")
        existing_data = pd.read_csv(filepath)

        if existing_data.empty or 'Date' not in existing_data.columns:
            print(f"Existing data file {filepath} is empty or missing 'Date' column.")
            return pd.DataFrame(), False  # Return a flag indicating that existing data is not found
        else:
            last_date = existing_data['Date'].iloc[-1]  # Get the last date in the existing data
            print(f"Found existing data up to {last_date} in {filepath}.")
            return existing_data, True  # Return a flag indicating that existing data is found
    except FileNotFoundError:
        print(f"Existing data file {filepath} not found.")
        return pd.DataFrame(), False  # Return a flag indicating that existing data is not found
    except Exception as e:
        print(f"Error reading existing data file {filepath}: {e}")
        return pd.DataFrame(), False  # Return a flag indicating that existing data is not found
    
async def fetch_calculate_update(input_files, output_files):
    for timeframe, filepath in output_files.items():
        print(f"Output file path for {timeframe}: {filepath}")
        
        existing_data, existing_data_found = read_existing_data(filepath)
        
        input_data = pd.read_csv(input_files[timeframe])
        
        if not existing_data_found:
            # If existing data is not found, delete the output file
            try:
                os.remove(filepath)
                print(f"Deleted {filepath} as existing data is not found.")
            except FileNotFoundError:
                pass
        
        new_data_available = False
        if not existing_data.empty:
            last_date = existing_data['Date'].iloc[-1]  # Assuming DATE column name
            print(f"Last date in existing data for {timeframe}: {last_date}")  # Debugging

            new_data = input_data[input_data['Date'] > last_date]  # Assuming DATE column name
            print(f"Found {len(new_data)} new records for {timeframe}.")  # Debugging

            if not new_data.empty:
                new_data_available = True
                merged_data = pd.concat([existing_data, new_data], ignore_index=True)
                print(f"Merged new data with existing data for {timeframe}.")  # Debugging

                if new_data_available or not os.path.exists(filepath):
                    print(f"Appending new data for {timeframe}...")
                    # Append the new data to the existing file
                    merged_data.to_csv(filepath, index=False, mode='a', header=not os.path.exists(filepath))
            else:
                print(f"No new data found for {timeframe}, skipping processing...")
        else:
            print(f"No existing data found for {timeframe}, processing all data...")
            # Process all data and save to the output file
            merged_data = process_data(timeframe, input_files[timeframe], filepath, existing_data)
            new_data_available = True

        if not new_data_available:
            print(f"No new data available for {timeframe}, skipping further processing...")

async def main():
    input_files = {'1h': '1h_crypto_data.csv', '12h': '1d_crypto_data.csv', '1d': '3d_crypto_data.csv', '3d': '1w_crypto_data.csv'}
    output_files = {'1h': '1h_Crypto_Monitor_List.csv', '12h': '1d_Crypto_Monitor_List.csv', '1d': '3d_Crypto_Monitor_List.csv', '3d': '1w_Crypto_Monitor_List.csv'}

 # Print output file paths
    for timeframe, filepath in output_files.items():
        print(f"Output file path for {timeframe}: {filepath}")

    await fetch_calculate_update(input_files, output_files)

# Run the function to start the event loop
print("Processing data...")
asyncio.run(main())

end_time = time.time()
duration = end_time - start_time
minutes = duration // 60
seconds = duration % 60

print("Data processing completed.")
print(f"Finished in {int(minutes)} minutes and {int(seconds)} seconds.")
