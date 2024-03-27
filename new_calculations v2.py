import pandas as pd
import talib
import numpy as np
import time
import requests



class TA:
    def __init__(self):
        # Initialize parameters for technical indicators
        self.MAfast = 10
        self.MAslow = 20
        self.MACDfast = 12
        self.MACDslow = 26
        self.MACDsignal = 9
        self.ATRperiod = 6
        self.ATRfactor = 2
        self.ADXperiod = 14
        self.ADXshift = 14
        self.DIperiod = 14
        self.Engulfperiod = 10
        self.RSIperiod = 14
        self.RSIupper = 70
        self.RSIlower = 30
        self.DemarkLookback = 10

    def MA(self, close):
        # Moving Average (MA) calculation
        output = close.copy()
        df = talib.MA(close, self.MAfast) - talib.MA(close, self.MAslow)
        df[df > 0] = 1
        df[df < 0] = -1
        return df
    
    def MACD_components(self, close):
        macd, macdsignal, macdhist = talib.MACD(close, self.MACDfast, self.MACDslow, self.MACDsignal)
        return macd, macdsignal, macdhist

    def MACD(self, close):
     macd, macdsignal, macdhist = talib.MACD(close, self.MACDfast, self.MACDslow, self.MACDsignal)
     score = pd.Series(0.0, index=close.index)
    
    # Assign scores based on MACD and signal line crossover and histogram size
     for i in range(1, len(close)):
        if macd[i] > macdsignal[i] and macd[i - 1] <= macdsignal[i - 1]:
            score[i] = 1  # Bullish crossover
        elif macd[i] < macdsignal[i] and macd[i - 1] >= macdsignal[i - 1]:
            score[i] = -1  # Bearish crossover
        
        # Enhance score based on histogram
        if macdhist[i] > macdhist[i - 1] and macdhist[i] > 0:
            score[i] += 0.5  # Strengthening bullish momentum
        elif macdhist[i] < macdhist[i - 1] and macdhist[i] < 0:
            score[i] -= 0.5  # Strengthening bearish momentum
     return score  # Corrected indentation to return outside the for loop
        
    def LL(self, high, low, close):
    # Initialize the score series with zeros
     score = pd.Series(0, index=close.index)

    # Bullish condition: today's close > yesterday's close and today's close > (high + low) / 2
    # Assign a positive score (for example, +1) to bullish signals
     score[(close > close.shift(1)) & (close > (high + low) / 2)] = 1

    # Bearish condition: today's close < yesterday's close and today's close < (high + low) / 2
    # Assign a negative score (for example, -1) to bearish signals
     score[(close < close.shift(1)) & (close < (high + low) / 2)] = -1

     return score


    def Trender(self, h, l, c):
        # Trend calculation based on ATR (Average True Range)
        atr = self.ATRfactor * talib.ATR(h, l, c, self.ATRperiod)
        up = 0.5 * (h + l) + atr
        down = up - 2 * atr
        st = down.copy()
        trend = down.copy()
        trend.iloc[:] = 1
        for i in range(1, c.shape[0]):
            if c.iloc[i] > st.iloc[i - 1] and trend.iloc[i - 1] < 0:
                trend.iloc[i] = 1
                st.iloc[i] = down.iloc[i]
            elif c.iloc[i] < st.iloc[i - 1] and trend.iloc[i - 1] > 0:
                trend.iloc[i] = -1
                st.iloc[i] = up.iloc[i]
            elif trend.iloc[i - 1] > 0:
                trend.iloc[i] = 1
                st.iloc[i] = max(down.iloc[i], st.iloc[i - 1])
            else:
                trend.iloc[i] = -1
                st.iloc[i] = min(up.iloc[i], st.iloc[i - 1])
        return trend

    def ADX(self, h, l, c):
        # Average Directional Movement Index (ADX) calculation
        adx = talib.ADX(h, l, c, self.ADXperiod)
        adx = (adx - adx.shift(self.ADXshift)) / 2
        adx[adx > 0] = 1
        adx[adx < 0] = -1
        return adx

    def DMI(self, h, l ,c):
        # Directional Movement Index (DMI) calculation
        pdi = talib.PLUS_DI(h, l, c, self.DIperiod)
        mdi = talib.MINUS_DI(h, l, c, self.DIperiod)
        dmi = pdi - mdi
        dmi[dmi > 0] = 1
        dmi[dmi < 0] = -1
        return dmi

    def Engulf_original(self, o, h, l, c):
        # Original engulfing pattern calculation
        black = c < o
        white = c > o
        engulf = (h > h.shift(1)) & (l < l.shift(1))
        bullEngulf = (white & black.shift(1) & engulf).astype(int)
        bearEngulf = (black & white.shift(1) & engulf).astype(int)
        sumBull = bullEngulf.rolling(self.Engulfperiod).sum()
        sumBear = bearEngulf.rolling(self.Engulfperiod).sum()
        netEngulf = 0.5 * (sumBull - sumBear)
        netEngulf[netEngulf >= 0.5] = 1
        netEngulf[netEngulf <= -0.5] = -1
        return netEngulf

    def Engulf(self, data, o, h, l, c):
        # Customized engulfing pattern calculation
        df = data[['OPEN', 'HIGH', 'LOW', 'CLOSE']].copy()
        black = c < o
        white = c > o
        engulf = (h > h.shift(1)) & (l < l.shift(1))
        bullEngulf = (white & black.shift(1) & engulf).astype(int)
        bearEngulf = (black & white.shift(1) & engulf).astype(int)

        df['bullEngulf'] = bullEngulf
        df['bullEngulfScore'] = 0
        df['bearEngulf'] = bearEngulf
        df['bearEngulfScore'] = 0
        df['engulf_score'] = 0

        for i in range(len(df)):
            # Calculate scores for bullish and bearish engulfing patterns
            if df['bullEngulf'][i] == 1:
                bull_high = df['HIGH'][i]
                bull_low = df['LOW'][i]
                for j in range(i + 1, min(i + 11, len(df))):
                    if df['CLOSE'][j] < bull_high and df['CLOSE'][j] > bull_low:
                        df.loc[df.index[j], 'bullEngulfScore'] = -1
                    if df['CLOSE'][j] > bull_high and df['CLOSE'][j] > df['CLOSE'][j - 1]:
                        df.loc[df.index[j], 'bullEngulfScore'] = -2
                    if df['bullEngulfScore'][j - 1] == -2 and df['bullEngulfScore'][j] != -2:
                        df.loc[df.index[j], 'bullEngulfScore'] = 0
                        break
                    if df['CLOSE'][j] < bull_low:
                        df.loc[df.index[j], 'bullEngulfScore'] = 0
                        break

            if df['bearEngulf'][i] == 1:
                bear_high = df['HIGH'][i]
                bear_low = df['LOW'][i]
                for j in range(i + 1, min(i + 11, len(df))):
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

            df['engulf_score'][i] = df['bullEngulfScore'][i] + df['bearEngulfScore'][i]

        engulf_score = df['engulf_score']
        return engulf_score

    def RSI(self, close):
        # Relative Strength Index (RSI) calculation
        rsi = talib.RSI(close, self.RSIperiod)
        rsi_ma = talib.MA(rsi, timeperiod=5)
        rsi_score = pd.Series(0, index=rsi.index)

        for i in range(1, len(rsi)):  # Start from 1 to avoid accessing -1 index
            # Modified conditions to correctly assign scores
            if rsi.iloc[i] > self.RSIupper or (rsi.iloc[i] > rsi_ma.iloc[i] and rsi.iloc[i - 1] <= rsi_ma.iloc[i - 1]):
                rsi_score.iloc[i] = 1
            elif rsi.iloc[i] < self.RSIlower or (rsi.iloc[i] < rsi_ma.iloc[i] and rsi.iloc[i - 1] >= rsi_ma.iloc[i - 1]):
                rsi_score.iloc[i] = -1
            # Added conditions to handle crossovers
            if rsi.iloc[i] > 70 and rsi.iloc[i] < rsi_ma.iloc[i] and rsi.iloc[i - 1] > rsi_ma.iloc[i - 1]:
                rsi_score.iloc[i] = 2
            elif rsi.iloc[i] < 30 and rsi.iloc[i] > rsi_ma.iloc[i] and rsi.iloc[i - 1] < rsi_ma.iloc[i - 1]:
                rsi_score.iloc[i] = -2
        return rsi_score


    def Demark(self, h, l, close):
        # Demark Indicator calculation
        s = close - close.shift(4)

        setup = close.copy()
        setup.iloc[0] = 0
        count = 0

        for i in range(1, close.shape[0]):
            # Counting setup trend
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

        countdown = close.copy()
        countdown.iloc[0] = 0
        count = 0

        for i in range(1, close.shape[0]):
            # Counting countdown trend
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
            # Calculating demark scores and trend lines
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

def process_data(timeframe, input_file, output_file):
    ta = TA()
    df = pd.read_csv(input_file)

# Renaming columns for consistency
    df.rename(columns={'Open': 'OPEN', 'High': 'HIGH', 'Low': 'LOW', 'Close': 'CLOSE'}, inplace=True)
    df['OPEN'] = df['OPEN'].astype(float)
    df['HIGH'] = df['HIGH'].astype(float)
    df['LOW'] = df['LOW'].astype(float)
    df['CLOSE'] = df['CLOSE'].astype(float)

    # Calculate technical indicators
    df['MA_Score'] = ta.MA(df['CLOSE'])
    df['MACD_Score'] = ta.MACD(df['CLOSE'])
    df['LL'] = ta.LL(df['HIGH'], df['LOW'], df['CLOSE'])
    df['Trender'] = ta.Trender(df['HIGH'], df['LOW'], df['CLOSE'])
    df['DMI'] = ta.DMI(df['HIGH'], df['LOW'], df['CLOSE'])
    df['Engulf'] = ta.Engulf(df, df['OPEN'], df['HIGH'], df['LOW'], df['CLOSE'])
    df['RSI_Score'] = ta.RSI(df['CLOSE'])
    df['Demark'] = ta.Demark(df['HIGH'], df['LOW'], df['CLOSE'])

     # Scores Signal Calculation
    df['Scores'] = df[['MA_Score', 'MACD_Score', 'LL', 'Trender', 'DMI', 'Engulf']].sum(axis=1)
    
    # Extreme Scores Signal Calculation
    df['Extreme Scores'] = df[['RSI_Score', 'LL', 'Engulf', 'Demark']].sum(axis=1)

    # Assign signals based on the score
    df['Signal'] = df['Scores'].apply(lambda x: assign_signal(x))

    # Assign extreme signals based on the score
    df['Extreme Signal'] = df['Extreme Scores'].apply(lambda x: assign_extreme_signal(x))
    df.rename(columns={'CLOSE': 'PRICE'}, inplace=True)

    df.to_csv(output_file, index=False)
    print(f"Processed data for {timeframe} saved to {output_file}")

    return df  # Return the processed dataframe
def assign_signal(score):
    if score >= 5:
        return 'Strong Buy'
    elif 3 <= score < 4:
        return 'Buy'
    elif score in [-2, 2]:
        return 'Neutral'
    elif -3 < score <= -4:
        return 'Sell'
    elif score <= -5:
        return 'Strong Sell'
    else:
        return 'Reversal'

def assign_extreme_signal(score):
    global last_extreme_signal
    if score >= 2.5:
        last_extreme_signal = 'Overbought'
        return 'Overbought'
    elif score <= -3:
        last_extreme_signal = 'Oversold'
        return 'Oversold'
    elif score == 0:
        if last_extreme_signal == 'Overbought':
            return 'Oversold'
        elif last_extreme_signal == 'Oversold':
            return 'Overbought'
        else:
            return 'Neutral'
    else:
        return 'Neutral'

def adjust_dataframe(df, timeframe):
    if timeframe == '1h':
        df = df.iloc[::3]
    elif timeframe == '12h':
        df = df.iloc[::2]
    elif timeframe in ['1d', '3d']:
        df = df.iloc[::3]
    return df

# Initialize the last extreme signal
last_extreme_signal = ''   

# Input and output file paths for different time frames
input_files = {'1h': '1h_crypto_data.csv', '12h': '1d_crypto_data.csv' , '1d': '3d_crypto_data.csv' ,'3d': '1w_crypto_data.csv'}
output_files = {'1h': '1h_Crypto_Monitor_List.csv', '12h': '1d_Crypto_Monitor_List.csv', '1d': '3d_Crypto_Monitor_List.csv' , '3d': '1w_Crypto_Monitor_List.csv'}

# Process data for each time frame
for timeframe, input_file in input_files.items():
    output_file = output_files[timeframe]
    process_data(timeframe, input_file, output_file)

print("Done")
