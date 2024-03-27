import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd

# Load the CSV files
daily_data = pd.read_csv('1d_Crypto_Monitor_List.csv')
hourly_data = pd.read_csv('1h_Crypto_Monitor_List.csv')
weekly_data = pd.read_csv('1w_Crypto_Monitor_List.csv')

# Convert 'Date' column to datetime if not already in datetime format
for data in [daily_data, hourly_data, weekly_data]: 
    data['Date'] = pd.to_datetime(data['Date'])

# Sort the data by date in descending order
for data in [daily_data, hourly_data, weekly_data]: 
    data.sort_values('Date', inplace=True, ascending=False)

# Get unique symbols
symbols = daily_data['Symbol'].unique()

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1('Cryptocurrency Data Visualization utc', style={'text-align': 'center'}),
    dcc.Dropdown(id='symbol-dropdown', options=[{'label': symbol, 'value': symbol} for symbol in symbols], value='BTC/USDT', style={'margin': 'auto', 'width': '50%'}),
    dcc.Dropdown(id='data-dropdown', options=[{'label': label, 'value': value} for label, value in [('Hourly Data', 'hourly'), ('Daily Data', 'daily'), ('Weekly Data', 'weekly')]], value='hourly', style={'margin': 'auto', 'width': '50%'}),
    html.Div([
        html.H1("Scores Signals"),
        dcc.Graph(id='crypto-graph', config={'displayModeBar': True}),
        html.H1("Extreme Scores Signals"),
        dcc.Graph(id='extreme-graph', config={'displayModeBar': True})
    ], style={'display': 'flex', 'flex-direction': 'column'}),

    html.H3("3 Hour Signal (Buy/Sell)", style={'text-align': 'center'}),
    dash_table.DataTable(
        id='data-table-1h', 
        columns=[
            {"name": i, "id": i} for i in hourly_data.columns
        ],
        data=hourly_data.to_dict('records'),
        page_action="native",
        page_size=10,
        filter_action="native",
        style_table={'margin': 'auto', 'width': '80%', 'maxWidth': '1200px'},
        style_cell={'textAlign': 'center'}
    ),
    html.H3("Daily Signal (Buy/Sell/Strong Buy/Strong Sell)", style={'text-align': 'center'}),
    dash_table.DataTable(
        id='data-table-daily', 
        columns=[
            {"name": i, "id": i} for i in daily_data.columns
        ],
        data=daily_data.to_dict('records'),
        page_action="native",
        page_size=10,
        filter_action="native",
        style_table={'margin': 'auto', 'width': '80%', 'maxWidth': '1200px'},
        style_cell={'textAlign': 'center'}
    ),
    html.H3("Weekly Signal (Buy/Sell/Strong Buy/Strong Sell)", style={'text-align': 'center'}),
    dash_table.DataTable(
        id='data-table-weekly', 
        columns=[
            {"name": i, "id": i} for i in weekly_data.columns
        ],
        data=weekly_data.to_dict('records'),
        page_action="native",
        page_size=10,
        filter_action="native",
        style_table={'margin': 'auto', 'width': '80%', 'maxWidth': '1200px'},
        style_cell={'textAlign': 'center'}
    )
])

@app.callback(
    [Output('crypto-graph', 'figure'),
     Output('extreme-graph', 'figure')],
    [Input('symbol-dropdown', 'value'),
     Input('data-dropdown', 'value')])
def update_graph(selected_symbol, selected_data):
    # Load the CSV files each time the callback is triggered
    daily_data = pd.read_csv('1d_Crypto_Monitor_List.csv')
    hourly_data = pd.read_csv('1h_Crypto_Monitor_List.csv')
    weekly_data = pd.read_csv('1w_Crypto_Monitor_List.csv')
    
    # Convert 'Date' column to datetime if not already in datetime format
    for data in [daily_data, hourly_data, weekly_data]: 
        data['Date'] = pd.to_datetime(data['Date'])

    # Sort the data by date in descending order
    for data in [daily_data, hourly_data, weekly_data]: 
        data.sort_values('Date', inplace=True, ascending=False)
    
    data_dict = {'hourly': hourly_data, 'daily': daily_data, 'weekly': weekly_data}
    data = data_dict[selected_data][data_dict[selected_data]['Symbol'] == selected_symbol]

    overbought_extreme_signals = data[data['Extreme Signal'] == 'Overbought']
    oversold_extreme_signals = data[data['Extreme Signal'] == 'Oversold']
    buy_signals = data[data['Signal'] == 'Buy']
    sell_signals = data[data['Signal'] == 'Sell']
    strong_sell_signals = data[data['Signal'] == 'Strong Sell']
    strong_buy_signals = data[data['Signal'] == 'Strong Buy']

    # Create the candlestick plot
    fig_candlestick = go.Figure()
    fig_candlestick.add_trace(go.Candlestick(x=data['Date'],
                                             open=data['OPEN'],
                                             high=data['HIGH'],
                                             low=data['LOW'],
                                             close=data['CLOSE'],
                                             increasing=dict(line=dict(color='green')),
                                             decreasing=dict(line=dict(color='red')),
                                             name='Candlestick'))

    # Add markers for Buy and Sell signals
    fig_candlestick.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals['CLOSE'], mode='markers', name='Buy Signal', marker=dict(color='green', symbol='x', size=15), showlegend=True))
    fig_candlestick.add_trace(go.Scatter(x=sell_signals['Date'], y=sell_signals['CLOSE'], mode='markers', name='Sell Signal', marker=dict(color='red', symbol='x', size=15), showlegend=True))

    fig_candlestick.add_trace(go.Scatter(x=strong_buy_signals['Date'], y=strong_buy_signals['CLOSE'], mode='lines', name='Strong Buy', line=dict(color='green', width=2), showlegend=True))
    fig_candlestick.add_trace(go.Scatter(x=strong_sell_signals['Date'], y=strong_sell_signals['CLOSE'], mode='lines', name='Strong Sell', line=dict(color='red', width=2), showlegend=True))

    # Add annotations for buy and sell signals

    for _, row in buy_signals.iterrows():
        fig_candlestick.add_annotation(x=row['Date'], y=row['CLOSE'], text='Buy', showarrow=True, arrowhead=1, ax=0, ay=-70)
    for _, row in sell_signals.iterrows():
        fig_candlestick.add_annotation(x=row['Date'], y=row['CLOSE'], text='Sell', showarrow=True, arrowhead=1, ax=0, ay=40)

    # Add annotations for strong buy and sell signals
    for _, row in strong_buy_signals.iterrows():
        fig_candlestick.add_annotation(x=row['Date'], y=row['CLOSE'], text='Strong Buy', showarrow=True, arrowhead=1, ax=0, ay=-100)
    for _, row in strong_sell_signals.iterrows():
        fig_candlestick.add_annotation(x=row['Date'], y=row['CLOSE'], text='Strong Sell', showarrow=True, arrowhead=1, ax=0, ay=100)

    fig_candlestick.update_layout(
        xaxis=dict(
            tickformat='%B %d, %Y %I:%M %p'  # Month, Day, Year Hour:Minute AM/PM
        ),
        title=f'{selected_symbol} Cryptocurrency Prices',
        xaxis_title='Date',
        yaxis_title='Price in USDT',
        height=800,
        margin=dict(t=0),
        title_font=dict(size=20, family='Arial', color='black'),
        title_x=0.5,
        updatemenus=[{'active': 0}],
    )

    # Create the extreme scores plot
    fig_extreme_candlestick = go.Figure()

    fig_extreme_candlestick.add_trace(go.Scatter(x=data['Date'], y=data['OPEN'], mode='lines', name='Open'))
    fig_extreme_candlestick.add_trace(go.Scatter(x=data['Date'], y=data['HIGH'], mode='lines', name='High'))
    fig_extreme_candlestick.add_trace(go.Scatter(x=data['Date'], y=data['LOW'], mode='lines', name='Low'))
    fig_extreme_candlestick.add_trace(go.Scatter(x=data['Date'], y=data['CLOSE'], mode='lines', name='Close'))

    fig_extreme_candlestick = go.Figure()
    fig_extreme_candlestick.add_trace(go.Candlestick(x=data['Date'],
                                                     open=data['OPEN'],
                                                     high=data['HIGH'],
                                                     low=data['LOW'],
                                                     close=data['CLOSE'],
                                                     increasing=dict(line=dict(color='green')),
                                                     decreasing=dict(line=dict(color='red')),
                                                     name='Candlestick'))
    
    fig_extreme_candlestick.add_trace(go.Scatter(x=oversold_extreme_signals['Date'], y=oversold_extreme_signals['CLOSE'], mode='markers', name='Oversold', marker=dict(color='red', symbol='triangle-up', size=20), showlegend=True))
    fig_extreme_candlestick.add_trace(go.Scatter(x=overbought_extreme_signals['Date'], y=overbought_extreme_signals['CLOSE'], mode='markers', name='Overbought', marker=dict(color='green', symbol='triangle-down', size=20), showlegend=True))
    fig_extreme_candlestick.add_trace(go.Scatter(x=overbought_extreme_signals['Date'], y=overbought_extreme_signals['CLOSE'], mode='lines', name='Overbought', line=dict(color='green', width=2), showlegend=True))
    fig_extreme_candlestick.add_trace(go.Scatter(x=oversold_extreme_signals['Date'], y=oversold_extreme_signals['CLOSE'], mode='lines', name='Oversold', line=dict(color='red', width=2), showlegend=True))

    for _, row in overbought_extreme_signals.iterrows():
        fig_extreme_candlestick.add_annotation(x=row['Date'], y=row['CLOSE'], text='Overbought', showarrow=True, arrowhead=1, ax=0, ay=-80)
   
    for _, row in oversold_extreme_signals.iterrows():
        fig_extreme_candlestick.add_annotation(x=row['Date'], y=row['CLOSE'], text='Oversold', showarrow=True, arrowhead=1, ax=0, ay=80)
    
    fig_extreme_candlestick.update_layout(
        xaxis=dict(
            tickformat='%B %d, %Y %I:%M %p'  # Month, Day, Year Hour:Minute AM/PM
        ),
        title=f'OHLC for {selected_symbol}',
        xaxis_title='Date',
        yaxis_title='Price in USDT',
        height=800,
        margin=dict(t=0),
        title_font=dict(size=20, family='Arial', color='black'),
        title_x=0.5,
        updatemenus=[{'active': 0}],
    )

    return fig_candlestick, fig_extreme_candlestick

if __name__ == '__main__':
    app.run_server(debug=True)
