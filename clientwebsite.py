import dash
from dash import dcc, html, dash_table
import plotly.graph_objs as go
import pandas as pd
from dash.dependencies import Input, Output
from pandas.errors import ParserError
import time

# Initialize the Dash app
app = dash.Dash(__name__)

app.title = 'Syrius Magic Monitor'

# Define the URL for the favicon
favicon_url = "favicon.ico"
# Set the favicon
app.index_string = f'<html><head><link rel="icon" type="image/x-icon" href="{favicon_url}"></head><body>' + app.index_string + '</body></html>'


def load_csv_data():
    try:
        hourly_data = pd.read_csv('1h_Crypto_Monitor_List.csv')
        daily_data = pd.read_csv('1d_Crypto_Monitor_List.csv')
        threeday_data = pd.read_csv('3d_Crypto_Monitor_List.csv')
        weekly_data = pd.read_csv('1w_Crypto_Monitor_List.csv')
    except ParserError as e:
        print("Error parsing CSV file:", e)
        return None, None, None, None

    # Convert 'Date' column to datetime if not already in datetime format
    for data in [daily_data, hourly_data, threeday_data, weekly_data]:
        data['Date'] = pd.to_datetime(data['Date'])

    # Sort the data by date in descending order
    for data in [daily_data, hourly_data, threeday_data, weekly_data]:
        data.sort_values('Date', inplace=True, ascending=False)

    return hourly_data, daily_data, threeday_data, weekly_data

# Load the CSV files initially
hourly_data, daily_data, threeday_data, weekly_data = load_csv_data()

# Get unique symbols
symbols = daily_data['Symbol'].unique()

# Define the layout
app.layout = html.Div([
    html.H1('Syrius Magic Monitor (Timezone:Asia/Singapore (GMT+8)', style={'text-align': 'center'}),
    dcc.Dropdown(id='symbol-dropdown', options=[{'label': symbol, 'value': symbol} for symbol in symbols], value='BTC/USDT', style={'margin': 'auto', 'width': '50%'}),
    dcc.Dropdown(id='data-dropdown', options=[{'label': label, 'value': value} for label, value in [('3 Hour Data', 'hourly'), ('Daily Data', 'daily'), ('3 Day Data', 'threeday'),('Weekly Data', 'weekly')]], value='hourly', style={'margin': 'auto', 'width': '50%'}),
    html.Button('Reload Data', id='reload-button', n_clicks=0, style={'margin': 'auto', 'display': 'block'}),
    html.Div([
        html.H1("Crypto New Signals Chart"),
        dcc.Graph(id='crypto-graph', config={'displayModeBar': True, 'responsive': True}),
        html.H1("Crypto Extreme Signals Chart"),
        dcc.Graph(id='extreme-graph', config={'displayModeBar': True, 'responsive': True})
    ], style={'display': 'flex', 'flex-direction': 'column'}),

    html.H3("3 Hour Signal (Buy/Sell/Strong Buy/Strong Sell)", style={'text-align': 'center'}),
    dash_table.DataTable(
        id='data-table-1h',
        columns=[{"name": i, "id": i} for i in hourly_data.columns if i in ['Date', 'Category', 'Symbol', 'Signal', 'OPEN', 'Extreme Signal', 'CURRENT PRICE']],  # Include 'Open' in columns
        data=hourly_data.assign(Date=lambda x: x['Date'].dt.strftime('%B %d, %Y %I:%M %p')).to_dict('records'),
        page_action="native",
        page_size=15,
        filter_action="native",
        style_table={'margin': 'auto', 'width': '100%', 'maxWidth': '1200px', 'overflowX': 'auto'},
        style_data_conditional=[{'if': {'filter_query': '{Signal} = "Sell"'},
                                 'backgroundColor': 'rgb(255, 165, 0)',
                                 'color': 'black'},
                                {'if': {'filter_query': '{Signal} = "Buy"'},
                                 'backgroundColor': 'rgb(144, 238, 144)',
                                 'color': 'black'},
                                {'if': {'filter_query': '{Signal} = "Strong Sell" or {Extreme Signal} = "Oversold"'},
                                 'backgroundColor': 'rgb(240,128,128)',
                                 'color': 'black'},
                                {'if': {'filter_query': '{Signal} = "Strong Buy" or {Extreme Signal} = "Overbought"'},
                                 'backgroundColor': 'rgb(154 , 205, 50)',
                                 'color': 'black'},
                                 {'if': {'filter_query': '{Extreme Signal} = "Reversal"'},
                                 'backgroundColor': 'rgb(211, 211, 211)',  # Light grey color
                                 'color': 'black'}],
        style_cell={'textAlign': 'center', 'minWidth': '100px', 'width': '100px', 'maxWidth': '100px', 'whiteSpace': 'normal', 'textOverflow': 'ellipsis'}
    ),
    html.H3("Daily Signal (Buy/Sell/Strong Buy/Strong Sell)", style={'text-align': 'center'}),
    dash_table.DataTable(
        id='data-table-daily',
        columns=[{"name": i, "id": i} for i in daily_data.columns if i in ['Date', 'Category', 'Symbol', 'Signal', 'OPEN', 'Extreme Signal', 'CURRENT PRICE']],  # Include 'Open' in columns
        data=daily_data.assign(Date=lambda x: x['Date'].dt.strftime('%B %d, %Y %I:%M %p')).to_dict('records'),
        filter_action="native",
        page_action="native",
        page_size=15,
        style_table={'margin': 'auto', 'width': '100%', 'maxWidth': '1200px', 'overflowX': 'auto'},
        style_data_conditional=[{'if': {'filter_query': '{Signal} = "Sell"'},
                                 'backgroundColor': 'rgb(255, 165, 0)',
                                 'color': 'black'},
                                {'if': {'filter_query': '{Signal} = "Buy"'},
                                 'backgroundColor': 'rgb(144, 238, 144)',
                                 'color': 'black'},
                                {'if': {'filter_query': '{Signal} = "Strong Sell" or {Extreme Signal} = "Oversold"'},
                                 'backgroundColor': 'rgb(240 ,128, 128)',
                                 'color': 'black'},
                                {'if': {'filter_query': '{Signal} = "Strong Buy" or {Extreme Signal} = "Overbought"'},
                                 'backgroundColor': 'rgb(154 ,205 ,50)',
                                 'color': 'black'},
                                 {'if': {'filter_query': '{Extreme Signal} = "Reversal"'},
                                 'backgroundColor': 'rgb(211, 211, 211)',  # Light grey color
                                 'color': 'black'}],
        style_cell={'textAlign': 'center', 'minWidth': '100px', 'width': '100px', 'maxWidth': '100px', 'whiteSpace': 'normal', 'textOverflow': 'ellipsis'}
    ),
    html.H3("3 Day Signal (Buy/Sell/Strong Buy/Strong Sell)", style={'text-align': 'center'}),
    dash_table.DataTable(
        id='data-table-threeday',
        columns=[{"name": i, "id": i} for i in threeday_data.columns if i in ['Date', 'Category', 'Symbol', 'Signal', 'OPEN', 'Extreme Signal', 'CURRENT PRICE']],  # Include 'Open' in columns
        data=threeday_data.assign(Date=lambda x: x['Date'].dt.strftime('%B %d, %Y %I:%M %p')).to_dict('records'),
        filter_action="native",
        page_action="native",
        page_size=15,
        style_table={'margin': 'auto', 'width': '100%', 'maxWidth': '1200px', 'overflowX': 'auto'},
        style_data_conditional=[{'if': {'filter_query': '{Signal} = "Sell"'},
                                 'backgroundColor': 'rgb(255, 165, 0)',
                                 'color': 'black'},
                                {'if': {'filter_query': '{Signal} = "Buy"'},
                                 'backgroundColor': 'rgb(144, 238, 144)',
                                 'color': 'black'},
                                {'if': {'filter_query': '{Signal} = "Strong Sell" or {Extreme Signal} = "Oversold"'},
                                 'backgroundColor': 'rgb(240 ,128, 128)',
                                 'color': 'black'},
                                {'if': {'filter_query': '{Signal} = "Strong Buy" or {Extreme Signal} = "Overbought"'},
                                 'backgroundColor': 'rgb(154 ,205 ,50)',
                                 'color': 'black'},
                                {'if': {'filter_query': '{Extreme Signal} = "Reversal"'},
                                 'backgroundColor': 'rgb(211, 211, 211)',  # Light grey color
                                 'color': 'black'}],
        style_cell={'textAlign': 'center', 'minWidth': '100px', 'width': '100px', 'maxWidth': '100px', 'whiteSpace': 'normal', 'textOverflow': 'ellipsis'}
    ),
    html.H3("Weekly Signal (Buy/Sell/Strong Buy/Strong Sell)", style={'text-align': 'center'}),
    dash_table.DataTable(
        id='data-table-weekly',
        columns=[{"name": i, "id": i} for i in weekly_data.columns if i in ['Date', 'Category', 'Symbol', 'Signal', 'OPEN', 'Extreme Signal', 'CURRENT PRICE']],  # Include 'Open' in columns
        data=weekly_data.assign(Date=lambda x: x['Date'].dt.strftime('%B %d, %Y %I:%M %p')).to_dict('records'),
        filter_action="native",
        page_action="native",
        page_size=15,
        style_table={'margin': 'auto', 'width': '100%', 'maxWidth': '1200px', 'overflowX': 'auto'},
        style_data_conditional=[{'if': {'filter_query': '{Signal} = "Sell"'},
                                 'backgroundColor': 'rgb(255, 165, 0)',
                                 'color': 'black'},
                                {'if': {'filter_query': '{Signal} = "Buy"'},
                                 'backgroundColor': 'rgb(144, 238, 144)',
                                 'color': 'black'},
                                {'if': {'filter_query': '{Signal} = "Strong Sell" or {Extreme Signal} = "Oversold"'},
                                 'backgroundColor': 'rgb(240 ,128 ,128)',
                                 'color': 'black'},
                                {'if': {'filter_query': '{Signal} = "Strong Buy" or {Extreme Signal} = "Overbought"'},
                                 'backgroundColor': 'rgb(154 ,205 ,50)',
                                 'color': 'black'},
                                {'if': {'filter_query': '{Extreme Signal} = "Reversal"'},
                                 'backgroundColor': 'rgb(211, 211, 211)',  # Light grey color
                                 'color': 'black'}],
        style_cell={'textAlign': 'center', 'minWidth': '100px', 'width': '100px', 'maxWidth': '100px', 'whiteSpace': 'normal', 'textOverflow': 'ellipsis'}
    ),
])


@app.callback(
    [Output('data-table-1h', 'data'),
     Output('data-table-daily', 'data'),
     Output('data-table-threeday', 'data'),
     Output('data-table-weekly', 'data')],
    [Input('symbol-dropdown', 'value'),
     Input('reload-button', 'n_clicks')]
)
def update_data(selected_symbol, n_clicks):
    global hourly_data, daily_data, threeday_data, weekly_data  # Declare as global
    if n_clicks > 0:
        # Reload CSV data
        hourly_data, daily_data, threeday_data, weekly_data = load_csv_data()

    # Filter data based on selected symbol
    filtered_hourly_data = hourly_data[hourly_data['Symbol'] == selected_symbol]
    filtered_daily_data = daily_data[daily_data['Symbol'] == selected_symbol]
    filtered_threeday_data = threeday_data[threeday_data['Symbol'] == selected_symbol]
    filtered_weekly_data = weekly_data[weekly_data['Symbol'] == selected_symbol]

    return (filtered_hourly_data.assign(Date=lambda x: x['Date'].dt.strftime('%B %d, %Y %I:%M %p')).to_dict('records'),
            filtered_daily_data.assign(Date=lambda x: x['Date'].dt.strftime('%B %d, %Y %I:%M %p')).to_dict('records'),
            filtered_threeday_data.assign(Date=lambda x: x['Date'].dt.strftime('%B %d, %Y %I:%M %p')).to_dict('records'),
            filtered_weekly_data.assign(Date=lambda x: x['Date'].dt.strftime('%B %d, %Y %I:%M %p')).to_dict('records'))


@app.callback(
    [Output('crypto-graph', 'figure'),
     Output('extreme-graph', 'figure')],
    [Input('symbol-dropdown', 'value'),
     Input('data-dropdown', 'value'),
     Input('reload-button', 'n_clicks')]
)
def update_graph(selected_symbol, selected_data, n_clicks):
    global hourly_data, daily_data, threeday_data, weekly_data  # Declare as global
    if n_clicks > 0:
        # Load the CSV files
        hourly_data, daily_data, threeday_data, weekly_data = load_csv_data()

    # Filter data based on selected symbol
    data_dict = {'hourly': hourly_data, 'daily': daily_data, 'threeday': threeday_data, 'weekly': weekly_data}
    data = data_dict[selected_data]
    data = data[data['Symbol'] == selected_symbol]

    filtered_data = data[(data['Signal'] != 'Neutral') | ((data['Extreme Signal'] != 'Neutral') & (data['Extreme Signal'] != 'Same'))]
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
                                             close=data['CURRENT PRICE'],
                                             increasing=dict(line=dict(color='green')),
                                             decreasing=dict(line=dict(color='red')),
                                             name='Candlestick'))

    fig_candlestick.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals['CURRENT PRICE'], mode='lines', name='Buy', line=dict(color='green', width=2), showlegend=True))
    fig_candlestick.add_trace(go.Scatter(x=sell_signals['Date'], y=sell_signals['CURRENT PRICE'], mode='lines', name='Sell', line=dict(color='red', width=2), showlegend=True))

        # Create legend
    fig_candlestick.update_layout(
    legend=dict(
        title='Signal',
        orientation='h',  # horizontal orientation
        yanchor='bottom',
        y=1.02,
        xanchor='right',
        x=1
    )
)

# Add annotations for buy and sell signals
    for _, row in buy_signals.iterrows():
     fig_candlestick.add_annotation(x=row['Date'], y=row['CURRENT PRICE'], text='Buy', showarrow=True, arrowhead=1, ax=0, ay=-40, textangle=-90, font=dict(size=10))
    for _, row in sell_signals.iterrows():
     fig_candlestick.add_annotation(x=row['Date'], y=row['CURRENT PRICE'], text='Sell', showarrow=True, arrowhead=1, ax=0, ay=60, textangle=90, font=dict(size=10))

# Add annotations for strong buy and sell signals
    #for _, row in strong_buy_signals.iterrows():
     #fig_candlestick.add_annotation(x=row['Date'], y=row['CURRENT PRICE'], text='Strong Buy', showarrow=True, arrowhead=1, ax=0, ay=-40 ,textangle=-90, font=dict(size=10))
    #for _, row in strong_sell_signals.iterrows():
     #fig_candlestick.add_annotation(x=row['Date'], y=row['CURRENT PRICE'], text='Strong Sell', showarrow=True, arrowhead=1, ax=0, ay=60 , textangle=90, font=dict(size=10))


    fig_candlestick.update_layout(
    xaxis=dict(
        tickformat='%B %d, %Y %I:%M %p',  # Month, Day, Year Hour:Minute AM/PM
        tickangle=-75
    ),
    title=f'{selected_symbol} Cryptocurrency Prices',
    xaxis_title='Date',
    yaxis_title='Price in USD',
    height=1000,
    margin=dict(t=1),
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
                                                     close=data['CURRENT PRICE'],
                                                     increasing=dict(line=dict(color='green')),
                                                     decreasing=dict(line=dict(color='red')),
                                                     name='Candlestick'))
    
    fig_extreme_candlestick.add_trace(go.Scatter(x=overbought_extreme_signals['Date'], y=overbought_extreme_signals['CURRENT PRICE'], mode='lines', name='Overbought', line=dict(color='green', width=2), showlegend=True))
    fig_extreme_candlestick.add_trace(go.Scatter(x=oversold_extreme_signals['Date'], y=oversold_extreme_signals['CURRENT PRICE'], mode='lines', name='Oversold', line=dict(color='red', width=2), showlegend=True))
    fig_extreme_candlestick.update_layout(
    legend=dict(
        title='Signal',
        orientation='h',  # horizontal orientation
        yanchor='bottom',
        y=1.02,
        xanchor='right',
        x=1
    )
)
    # Plot overbought signals
    for _, row in overbought_extreme_signals.iterrows():
        fig_extreme_candlestick.add_annotation(x=row['Date'], y=row['CURRENT PRICE'], text='Overbought', showarrow=True, arrowhead=1, ax=0, ay=-40, textangle=-90 ,font=dict(size=10))
    # Plot oversold signals
    for _, row in oversold_extreme_signals.iterrows():
        fig_extreme_candlestick.add_annotation(x=row['Date'], y=row['CURRENT PRICE'], text='Oversold', showarrow=True, arrowhead=1, ax=0, ay=60 ,textangle=90 ,font=dict(size=10))

    fig_extreme_candlestick.update_layout(
    xaxis=dict(
        tickformat='%B %d, %Y %I:%M %p',  # Month, Day, Year Hour:Minute AM/PM
        tickangle=-75
    ),
    title=f'{selected_symbol} Cryptocurrency Prices',
    xaxis_title='Date',
    yaxis_title='Price in USD',
    height=1000,
    margin=dict(t=1),
    title_font=dict(size=20, family='Arial', color='black'),
    title_x=0.5,
    updatemenus=[{'active': 0}],
)

    return fig_candlestick, fig_extreme_candlestick

if __name__ == '__main__':
    app.run_server(debug=False, host='127.0.0.1', port=8030)
