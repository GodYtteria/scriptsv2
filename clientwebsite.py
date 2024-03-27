import dash
from dash import dcc, html, dash_table, Input, Output
import plotly.graph_objs as go
import pandas as pd
from flask_caching import Cache
import os

# Initialize the Dash app
app = dash.Dash(__name__)

# Configure cache
cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',  # Use 'simple' for in-memory caching
    'CACHE_DIR': 'cache-directory',  # Specify the directory to store cache files if using filesystem cache
    'CACHE_THRESHOLD': 5000  # The maximum number of items the cache will store
})

CACHE_TIMEOUT = 60  # Time in seconds for cache to refresh

def load_csv_data():
    # Check if CSV data is in cache
    cached_data = cache.get('csv_data')
    if cached_data is not None:
        return cached_data
    else:
        # Load the CSV files
        hourly_data = pd.read_csv('1h_Crypto_Monitor_List.csv')
        daily_data = pd.read_csv('1d_Crypto_Monitor_List.csv')
        threeday_data = pd.read_csv('3d_Crypto_Monitor_List.csv')
        weekly_data = pd.read_csv('1w_Crypto_Monitor_List.csv')

        # Convert 'Date' column to datetime if not already in datetime format
        for data in [daily_data, hourly_data, threeday_data, weekly_data]: 
            data['Date'] = pd.to_datetime(data['Date'])

        # Sort the data by date in descending order
        for data in [daily_data, hourly_data, threeday_data, weekly_data]:
            data.sort_values('Date', inplace=True, ascending=False)

        # Cache the loaded data
        cache.set('csv_data', (hourly_data, daily_data, threeday_data, weekly_data), timeout=CACHE_TIMEOUT)
        
        return hourly_data, daily_data, threeday_data, weekly_data

# Replace direct CSV loading in your app with a call to load_csv_data()
hourly_data, daily_data, threeday_data, weekly_data = load_csv_data()

# Get unique symbols
symbols = daily_data['Symbol'].unique()

# Define the layout
app.layout = html.Div([
    html.H1('Cryptocurrency Data Visualization (UTC)', style={'text-align': 'center'}),
    dcc.Dropdown(id='symbol-dropdown', options=[{'label': symbol, 'value': symbol} for symbol in symbols], value='BTC/USDT', style={'margin': 'auto', 'width': '50%'}),
    dcc.Dropdown(id='data-dropdown', options=[{'label': label, 'value': value} for label, value in [('3 Hour Data', 'hourly'), ('Daily Data', 'daily'), ('3 Day Data', 'threeday'),('Weekly Data', 'weekly')]], value='hourly', style={'margin': 'auto', 'width': '50%'}),
    html.Div([
    html.H1("Scores Signals"),
    dcc.Graph(id='crypto-graph', config={'displayModeBar': True}),
    html.H1("Extreme Scores Signals"),
    dcc.Graph(id='extreme-graph', config={'displayModeBar': True})
], style={'display': 'flex', 'flex-direction': 'column'}),

    html.H3("3 Hour Signal (Buy/Sell/Strong Buy/Strong Sell)", style={'text-align': 'center'}),
    dash_table.DataTable(id='data-table-1h', columns=[{"name": i, "id": i} for i in hourly_data.columns if i in ['Date', 'Category', 'Symbol','Signal','Signal', 'Extreme Signal', 'PRICE']], data=hourly_data.assign(Date=lambda x: x['Date'].dt.strftime('%B %d, %Y %I:%M %p')).to_dict('records')
, page_action="native", page_size=10, filter_action="native", style_table={'margin': 'auto', 'width': '80%', 'maxWidth': '1200px'}, style_data_conditional=[{'if': {'filter_query': '{Signal} = "Sell"'},'backgroundColor': 'rgb(255, 165, 0)','color': 'black'},{'if': {'filter_query': '{Signal} = "Buy"'},
 'backgroundColor': 'rgb(144, 238, 144)','color': 'black'},{'if': {'filter_query': '{Signal} = "Strong Sell" or {Extreme Signal} = "Oversold"'},
 'backgroundColor': 'rgb(255, 0, 0)','color': 'black'},{'if': {'filter_query': '{Signal} = "Strong Buy" or {Extreme Signal} = "Overbought"'},
 'backgroundColor': 'rgb(0, 128, 0)','color': 'black'}], style_cell={'textAlign': 'center'}),
    html.H3("Daily Signal (Buy/Sell/Strong Buy/Strong Sell)", style={'text-align': 'center'}),
    dash_table.DataTable(id='data-table-daily', columns=[{"name": i, "id": i} for i in 
    daily_data.columns if i in ['Date', 'Category', 'Symbol', 'Signal', 'Extreme Signal', 'PRICE']], 
    data=daily_data.assign(Date=lambda x: x['Date'].dt.strftime('%B %d, %Y %I:%M %p')).to_dict('records'),
    filter_action="native", page_action="native", page_size=10, style_table={'margin': 'auto', 'width': '80%', 'maxWidth': '1200px'}, style_data_conditional=[{'if': {'filter_query': '{Signal} = "Sell"'},'backgroundColor': 'rgb(255, 165, 0)','color': 'black'},{'if': {'filter_query': '{Signal} = "Buy"'},
 'backgroundColor': 'rgb(144, 238, 144)','color': 'black'},{'if': {'filter_query': '{Signal} = "Strong Sell" or {Extreme Signal} = "Oversold"'},
 'backgroundColor': 'rgb(255, 0, 0)','color': 'black'},{'if': {'filter_query': '{Signal} = "Strong Buy" or {Extreme Signal} = "Overbought"'},
 'backgroundColor': 'rgb(0, 128, 0)','color': 'black'}], style_cell={'textAlign': 'center'}),
 html.H3("3 Day Signal (Buy/Sell/Strong Buy/Strong Sell)", style={'text-align': 'center'}),
    dash_table.DataTable(id='data-table-threeday', columns=[{"name": i, "id": i} for i in threeday_data.columns if i in ['Date', 'Category', 'Symbol', 'Signal', 'Extreme Signal', 'PRICE']], data=threeday_data.assign(Date=lambda x: x['Date'].dt.strftime('%B %d, %Y %I:%M %p')).to_dict('records'), filter_action="native", page_action="native", page_size=10, style_table={'margin': 'auto', 'width': '80%', 'maxWidth': '1200px'}, style_data_conditional=[{'if': {'filter_query': '{Signal} = "Sell"'},'backgroundColor': 'rgb(255, 165, 0)','color': 'black'},{'if': {'filter_query': '{Signal} = "Buy"'},
 'backgroundColor': 'rgb(144, 238, 144)','color': 'black'},{'if': {'filter_query': '{Signal} = "Strong Sell" or {Extreme Signal} = "Oversold"'},
 'backgroundColor': 'rgb(255, 0, 0)','color': 'black'},{'if': {'filter_query': '{Signal} = "Strong Buy" or {Extreme Signal} = "Overbought"'},
 'backgroundColor': 'rgb(0, 128, 0)','color': 'black'}], style_cell={'textAlign': 'center'}),
    html.H3("Weekly Signal (Buy/Sell/Strong Buy/Strong Sell)", style={'text-align': 'center'}),
    dash_table.DataTable(id='data-table-weekly', columns=[{"name": i, "id": i} for i in weekly_data.columns if i in ['Date', 'Category', 'Symbol', 'Signal', 'Extreme Signal', 'PRICE']], data=weekly_data.assign(Date=lambda x: x['Date'].dt.strftime('%B %d, %Y %I:%M %p')).to_dict('records'), filter_action="native", page_action="native", page_size=10, style_table={'margin': 'auto', 'width': '80%', 'maxWidth': '1200px'}, style_data_conditional=[{'if': {'filter_query': '{Signal} = "Sell"'},'backgroundColor': 'rgb(255, 165, 0)','color': 'black'},{'if': {'filter_query': '{Signal} = "Buy"'},
 'backgroundColor': 'rgb(144, 238, 144)','color': 'black'},{'if': {'filter_query': '{Signal} = "Strong Sell" or {Extreme Signal} = "Oversold"'},
 'backgroundColor': 'rgb(255, 0, 0)','color': 'black'},{'if': {'filter_query': '{Signal} = "Strong Buy" or {Extreme Signal} = "Overbought"'},
 'backgroundColor': 'rgb(0, 128, 0)','color': 'black'}], style_cell={'textAlign': 'center'})
])

@app.callback(
    [Output('crypto-graph', 'figure'),
     Output('extreme-graph', 'figure')],
    [Input('symbol-dropdown', 'value'),
     Input('data-dropdown', 'value')])

def update_graph(selected_symbol, selected_data):
    # Filter data based on selected symbol
    data_dict = {'hourly': hourly_data, 'daily': daily_data, 'threeday': threeday_data, 'weekly': weekly_data}
    data = data_dict[selected_data]
    data = data[data['Symbol'] == selected_symbol]
    
    def alternate_signals(data, signal_column='Signal'):
        last_signal = None
        alternating_rows = []

        for index, row in data.iterrows():
            current_signal = row[signal_column]
            # Check if the current signal is different from the last signal and is not 'Neutral' or 'Same'
            if current_signal not in ['Neutral', 'Same'] and current_signal != last_signal:
                alternating_rows.append(row)
                last_signal = current_signal

        return pd.DataFrame(alternating_rows)

    # Apply alternation logic separately for main and extreme signals
    alternating_data = alternate_signals(data, 'Signal')
    alternating_extreme_data = alternate_signals(data, 'Extreme Signal')

    # From here, you can filter out specific signals like 'Overbought' or 'Oversold' from alternating_extreme_data
    # and other signal types from alternating_data as needed.

    filtered_data = data[(data['Signal'] != 'Neutral') & (data['Signal'] != 'Same') | ((data['Extreme Signal'] != 'Neutral') & (data['Extreme Signal'] != 'Same'))]
    overbought_extreme_signals = filtered_data[filtered_data['Extreme Signal'] == 'Overbought']
    oversold_extreme_signals = filtered_data[filtered_data['Extreme Signal'] == 'Oversold']
    buy_signals = filtered_data[filtered_data['Signal'] == 'Buy']
    sell_signals = filtered_data[filtered_data['Signal'] == 'Sell']
    strong_sell_signals = filtered_data[filtered_data['Signal'] == 'Strong Sell']
    strong_buy_signals = filtered_data[filtered_data['Signal'] == 'Strong Buy']

    # Create the candlestick plot
    fig_candlestick = go.Figure()
    fig_candlestick.add_trace(go.Candlestick(x=data['Date'],
                                             open=data['OPEN'],
                                             high=data['HIGH'],
                                             low=data['LOW'],
                                             close=data['PRICE'],
                                             increasing=dict(line=dict(color='green')),
                                             decreasing=dict(line=dict(color='red')),
                                             name='Candlestick'))

    # Add markers for Buy and Sell signals
    # Initialize variables to keep track of the previous signal type and its date
    prev_signal = None
    prev_date = None

# Plot buy and sell signals alternatively
    for date, signal, price in zip(buy_signals['Date'], buy_signals['Signal'], buy_signals['PRICE']):
     if signal != prev_signal or prev_date is None or (date - prev_date).days > 1:
        fig_candlestick.add_trace(go.Scatter(x=[date], y=[price], mode='markers', name='Buy Signal', marker=dict(color='green', symbol='x', size=15), showlegend=True))
        prev_signal = signal
        prev_date = date

    for date, signal, price in zip(sell_signals['Date'], sell_signals['Signal'], sell_signals['PRICE']):
     if signal != prev_signal or prev_date is None or (date - prev_date).days > 1:
        fig_candlestick.add_trace(go.Scatter(x=[date], y=[price], mode='markers', name='Sell Signal', marker=dict(color='red', symbol='x', size=15), showlegend=True))
        prev_signal = signal
        prev_date = date

    fig_candlestick.add_trace(go.Scatter(x=strong_buy_signals['Date'], y=strong_buy_signals['PRICE'], mode='lines', name='Strong Buy', line=dict(color='green', width=2), showlegend=True))
    fig_candlestick.add_trace(go.Scatter(x=strong_sell_signals['Date'], y=strong_sell_signals['PRICE'], mode='lines', name='Strong Sell', line=dict(color='red', width=2), showlegend=True))

    # Add annotations for buy and sell signals
    for _, row in buy_signals.iterrows():
        fig_candlestick.add_annotation(x=row['Date'], y=row['PRICE'], text='Buy', showarrow=True, arrowhead=1, ax=0, ay=-40)
    for _, row in sell_signals.iterrows():
        fig_candlestick.add_annotation(x=row['Date'], y=row['PRICE'], text='Sell', showarrow=True, arrowhead=1, ax=0, ay=40)

    # Add annotations for strong buy and sell signals
    for _, row in strong_buy_signals.iterrows():
        fig_candlestick.add_annotation(x=row['Date'], y=row['PRICE'], text='Strong Buy', showarrow=True, arrowhead=1, ax=0, ay=-100)
    for _, row in strong_sell_signals.iterrows():
        fig_candlestick.add_annotation(x=row['Date'], y=row['PRICE'], text='Strong Sell', showarrow=True, arrowhead=1, ax=0, ay=100)

    fig_candlestick.update_layout(
        xaxis=dict(
            tickformat='%B %d, %Y %I:%M %p'  # Month, Day, Year Hour:Minute AM/PM
        ),
        title=f'{selected_symbol} Cryptocurrency Prices',
        xaxis_title='Date',
        yaxis_title='Price in USDT',
        height=700,
        margin=dict(t=0),
        title_font=dict(size=20, family='Arial', color='black'),
        title_x=0.5,
        updatemenus=[{'active': 0}],
    )

    # Create the extreme scores plot
    fig_extreme_candlestick = go.Figure()

    fig_extreme_candlestick.add_trace(go.Candlestick(x=data['Date'],
                                                     open=data['OPEN'],
                                                     high=data['HIGH'],
                                                     low=data['LOW'],
                                                     close=data['PRICE'],
                                                     increasing=dict(line=dict(color='green')),
                                                     decreasing=dict(line=dict(color='red')),
                                                     name='Candlestick'))
    
    fig_extreme_candlestick.add_trace(go.Scatter(x=oversold_extreme_signals['Date'], y=oversold_extreme_signals['PRICE'], mode='markers', name='Oversold', marker=dict(color='red', symbol='triangle-down', size=20), showlegend=True))
    fig_extreme_candlestick.add_trace(go.Scatter(x=overbought_extreme_signals['Date'], y=overbought_extreme_signals['PRICE'], mode='markers', name='Overbought', marker=dict(color='green', symbol='triangle-up', size=20), showlegend=True))
    fig_extreme_candlestick.add_trace(go.Scatter(x=overbought_extreme_signals['Date'], y=overbought_extreme_signals['PRICE'], mode='lines', name='Overbought', line=dict(color='green', width=2), showlegend=True))
    fig_extreme_candlestick.add_trace(go.Scatter(x=oversold_extreme_signals['Date'], y=oversold_extreme_signals['PRICE'], mode='lines', name='Oversold', line=dict(color='red', width=2), showlegend=True))

    # Annotate overbought and oversold signals
    for _, row in overbought_extreme_signals.iterrows():
     fig_extreme_candlestick.add_annotation(x=row['Date'], y=row['PRICE'], text='Overbought', showarrow=True, arrowhead=1, ax=0, ay=-80)
   
    for _, row in oversold_extreme_signals.iterrows():
        fig_extreme_candlestick.add_annotation(x=row['Date'], y=row['PRICE'], text='Oversold', showarrow=True, arrowhead=1, ax=0, ay=80)
    
    fig_extreme_candlestick.update_layout(
        xaxis=dict(
            tickformat='%B %d, %Y %I:%M %p'  # Month, Day, Year Hour:Minute AM/PM
        ),
        title=f'OHLC for {selected_symbol}',
        xaxis_title='Date',
        yaxis_title='Price in USDT',
        height=700,
        margin=dict(t=0),
        title_font=dict(size=20, family='Arial', color='black'),
        title_x=0.5,
        updatemenus=[{'active': 0}],
    )

    return fig_candlestick, fig_extreme_candlestick

if __name__ == '__main__':
    app.run_server(debug=True)

