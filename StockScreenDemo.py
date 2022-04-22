# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 17:45:01 2021

@author: Teo Bee Guan
"""

import pandas as pd
import datetime
from pandas_datareader import data as pdr
import investpy
import yfinance as yf
import streamlit as st
import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.sidebar.image("https://slm-sa.com/wp-content/uploads/2020/01/LogoSLM-IF-1.png")
st.sidebar.write("""
# RIVACUBE SCREENER 
## First Jet for simple analysis
*Built by SLM for Rivaldi Project*
""")

snp500 = pd.read_csv("Datasets/SP500.csv")
symbols = snp500['Symbol'].sort_values().tolist()        


ticker = st.sidebar.selectbox(
    'Choose a S&P 500 Stock',
     symbols)

traday = pdr.DataReader(ticker, data_source="yahoo")



infoType = st.sidebar.radio(
        "Choose an info type",
        ('Fundamental', 'Technical', 'Intraday')
    )

stock = yf.Ticker(ticker)

if(infoType == 'Fundamental'):
    stock = yf.Ticker(ticker)
    info = stock.info 
    st.title('Company Profile')
    st.subheader(info['longName']) 
    st.markdown('** Sector **: ' + info['sector'])
    st.markdown('** Industry **: ' + info['industry'])
    st.markdown('** Phone **: ' + info['phone'])
    st.markdown('** Address **: ' + info['address1'] + ', ' + info['city'] + ', ' + info['zip'] + ', '  +  info['country'])
    st.markdown('** Website **: ' + info['website'])
    st.markdown('** Business Summary **')
    st.info(info['longBusinessSummary'])
        
    fundInfo = {
            'Enterprise Value (USD)': info['enterpriseValue'],
            'Enterprise To Revenue Ratio': info['enterpriseToRevenue'],
            'Enterprise To Ebitda Ratio': info['enterpriseToEbitda'],
            'Net Income (USD)': info['netIncomeToCommon'],
            'Profit Margin Ratio': info['profitMargins'],
            'Forward PE Ratio': info['forwardPE'],
            'PEG Ratio': info['pegRatio'],
            'Price to Book Ratio': info['priceToBook'],
            'Forward EPS (USD)': info['forwardEps'],
            'Beta ': info['beta'],
            'Book Value (USD)': info['bookValue'],
            'Dividend Rate (%)': info['dividendRate'], 
            'Dividend Yield (%)': info['dividendYield'],
            'Five year Avg Dividend Yield (%)': info['fiveYearAvgDividendYield'],
            'Payout Ratio': info['payoutRatio']
        }
    
    fundDF = pd.DataFrame.from_dict(fundInfo, orient='index')
    fundDF = fundDF.rename(columns={0: 'Value'})
    st.subheader('Fundamental Info') 
    st.table(fundDF)
    
    st.subheader('General Stock Info') 
    st.markdown('** Market **: ' + info['market'])
    st.markdown('** Exchange **: ' + info['exchange'])
    st.markdown('** Quote Type **: ' + info['quoteType'])
    
    start = dt.datetime.today()-dt.timedelta(2 * 365)
    end = dt.datetime.today()
    df = yf.download(ticker,start,end)
    df = df.reset_index()
    fig = go.Figure(
            data=go.Scatter(x=df['Date'], y=df['Adj Close'])
        )
    fig.update_layout(
        title={
            'text': "Stock Prices Over Past Two Years",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
    st.plotly_chart(fig, use_container_width=True)
    
    marketInfo = {
            "Volume": info['volume'],
            "Average Volume": info['averageVolume'],
            "Market Cap": info["marketCap"],
            "Float Shares": info['floatShares'],
            "Regular Market Price (USD)": info['regularMarketPrice'],
            'Bid Size': info['bidSize'],
            'Ask Size': info['askSize'],
            "Share Short": info['sharesShort'],
            'Short Ratio': info['shortRatio'],
            'Share Outstanding': info['sharesOutstanding']
    
        }
    
    marketDF = pd.DataFrame(data=marketInfo, index=[0])
    st.table(marketDF)
if(infoType == 'Technical'):
    def calcMovingAverage(data, size):
        df = data.copy()
        df['sma'] = df['Adj Close'].rolling(size).mean()
        df['ema'] = df['Adj Close'].ewm(span=size, min_periods=size).mean()
        df.dropna(inplace=True)
        return df
    
    def calc_macd(data):
        df = data.copy()
        df['ema12'] = df['Adj Close'].ewm(span=12, min_periods=12).mean()
        df['ema26'] = df['Adj Close'].ewm(span=26, min_periods=26).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['signal'] = df['macd'].ewm(span=9, min_periods=9).mean()
        df.dropna(inplace=True)
        return df

    def calcBollinger(data, size):
        df = data.copy()
        df["sma"] = df['Adj Close'].rolling(size).mean()
        df["bolu"] = df["sma"] + 2*df['Adj Close'].rolling(size).std(ddof=0) 
        df["bold"] = df["sma"] - 2*df['Adj Close'].rolling(size).std(ddof=0) 
        df["width"] = df["bolu"] - df["bold"]
        df.dropna(inplace=True)
        return df

    st.title('Technical Indicators')
    st.subheader('Moving Average')
    
    coMA1, coMA2 = st.beta_columns(2)
    
    with coMA1:
        numYearMA = st.number_input('Insert period (Year): ', min_value=1, max_value=10, value=2, key=0)    
    
    with coMA2:
        windowSizeMA = st.number_input('Window Size (Day): ', min_value=5, max_value=500, value=20, key=1)  
        

    start = dt.datetime.today()-dt.timedelta(numYearMA * 365)
    end = dt.datetime.today()
    dataMA = yf.download(ticker,start,end)
    df_ma = calcMovingAverage(dataMA, windowSizeMA)
    df_ma = df_ma.reset_index()
        
    figMA = go.Figure()
    
    figMA.add_trace(
            go.Scatter(
                    x = df_ma['Date'],
                    y = df_ma['Adj Close'],
                    name = "Prices Over Last " + str(numYearMA) + " Year(s)"
                )
        )
    
    figMA.add_trace(
                go.Scatter(
                        x = df_ma['Date'],
                        y = df_ma['sma'],
                        name = "SMA" + str(windowSizeMA) + " Over Last " + str(numYearMA) + " Year(s)"
                    )
            )
    
    figMA.add_trace(
                go.Scatter(
                        x = df_ma['Date'],
                        y = df_ma['ema'],
                        name = "EMA" + str(windowSizeMA) + " Over Last " + str(numYearMA) + " Year(s)"
                    )
            )
    
    figMA.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))
    
    figMA.update_layout(legend_title_text='Trend')
    figMA.update_yaxes(tickprefix="$")
    
    st.plotly_chart(figMA, use_container_width=True)  
    
    st.subheader('Moving Average Convergence Divergence (MACD)')
    numYearMACD = st.number_input('Insert period (Year): ', min_value=1, max_value=10, value=2, key=2) 
    
    startMACD = dt.datetime.today()-dt.timedelta(numYearMACD * 365)
    endMACD = dt.datetime.today()
    dataMACD = yf.download(ticker,startMACD,endMACD)
    df_macd = calc_macd(dataMACD)
    df_macd = df_macd.reset_index()
    
    figMACD = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.01)
    
    figMACD.add_trace(
            go.Scatter(
                    x = df_macd['Date'],
                    y = df_macd['Adj Close'],
                    name = "Prices Over Last " + str(numYearMACD) + " Year(s)"
                ),
            row=1, col=1
        )
    
    figMACD.add_trace(
            go.Scatter(
                    x = df_macd['Date'],
                    y = df_macd['ema12'],
                    name = "EMA 12 Over Last " + str(numYearMACD) + " Year(s)"
                ),
            row=1, col=1
        )
    
    figMACD.add_trace(
            go.Scatter(
                    x = df_macd['Date'],
                    y = df_macd['ema26'],
                    name = "EMA 26 Over Last " + str(numYearMACD) + " Year(s)"
                ),
            row=1, col=1
        )
    
    figMACD.add_trace(
            go.Scatter(
                    x = df_macd['Date'],
                    y = df_macd['macd'],
                    name = "MACD Line"
                ),
            row=2, col=1
        )
    
    figMACD.add_trace(
            go.Scatter(
                    x = df_macd['Date'],
                    y = df_macd['signal'],
                    name = "Signal Line"
                ),
            row=2, col=1
        )
    
    figMACD.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1,
        xanchor="left",
        x=0
    ))
    
    figMACD.update_yaxes(tickprefix="$")
    st.plotly_chart(figMACD, use_container_width=True)
    
    st.subheader('Bollinger Band')
    coBoll1, coBoll2 = st.beta_columns(2)
    with coBoll1:
        numYearBoll = st.number_input('Insert period (Year): ', min_value=1, max_value=10, value=2, key=6) 
        
    with coBoll2:
        windowSizeBoll = st.number_input('Window Size (Day): ', min_value=5, max_value=500, value=20, key=7)
    
    startBoll= dt.datetime.today()-dt.timedelta(numYearBoll * 365)
    endBoll = dt.datetime.today()
    dataBoll = yf.download(ticker,startBoll,endBoll)
    df_boll = calcBollinger(dataBoll, windowSizeBoll)
    df_boll = df_boll.reset_index()
    figBoll = go.Figure()
    figBoll.add_trace(
            go.Scatter(
                    x = df_boll['Date'],
                    y = df_boll['bolu'],
                    name = "Upper Band"
                )
        )
    
    
    figBoll.add_trace(
                go.Scatter(
                        x = df_boll['Date'],
                        y = df_boll['sma'],
                        name = "SMA" + str(windowSizeBoll) + " Over Last " + str(numYearBoll) + " Year(s)"
                    )
            )
    
    
    figBoll.add_trace(
                go.Scatter(
                        x = df_boll['Date'],
                        y = df_boll['bold'],
                        name = "Lower Band"
                    )
            )
    
    figBoll.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1,
        xanchor="left",
        x=0
    ))
    
    figBoll.update_yaxes(tickprefix="$")
    st.plotly_chart(figBoll, use_container_width=True)
if(infoType == 'Intraday'):
    d1 = st.date_input(
     "Choose the beginning date",
     datetime.date(2022, 4, 5))
    st.write('From:', d1)
    d2 = st.date_input(
     "Choose the end date",
     datetime.date(2022, 4, 7))
    st.write('To:', d2)

    stock = yf.Ticker(ticker)

    Intraday = pdr.DataReader(ticker, data_source="yahoo",start=d1,end=d2)
    st.table(Intraday)
    def convert_df(df):
        return df.to_csv().encode('utf-8')


    csv = convert_df(Intraday)
    st.download_button(
   "Press to Download",
   csv,
   "file.csv",
   "text/csv",
   key='download-csv'
    )
