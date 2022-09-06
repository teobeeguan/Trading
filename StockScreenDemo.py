# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 17:45:01 2021

Credits to Original @author: Teo Bee Guan
"""

import pandas as pd
import datetime
from pandas_datareader import data as pdr
import investpy
import yfinance as yf
import streamlit as st
import base64
import streamlit.components.v1 as components
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from random import randint 
import plotly.express as px
import json # for graph plotting in website
# NLTK VADER for sentiment analysis
import nltk
nltk.downloader.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px
from prophet import Prophet
from plotly import graph_objs as go
from prophet.plot import plot_plotly



st.sidebar.image("https://slm-sa.com/wp-content/uploads/2020/01/LogoSLM-IF-1.png")
st.sidebar.write("""
# RIVACUBE SCREENER 
## Dashboard for Data Visualization
*By SLM for Rivaldi Project*

""")

snp500 = pd.read_csv("Datasets/SP500.csv")
symbols = snp500['Symbol'].sort_values().tolist()        
# for extracting data from finviz
finviz_url = 'https://finviz.com/quote.ashx?t='

ticker = st.sidebar.selectbox(
    'Choose a S&P 500 Stock',
     symbols)

traday = pdr.DataReader(ticker, data_source="yahoo")



infoType = st.sidebar.radio(
        "Choose an analysis type",
        ('Fundamental', 'Technical', 'Intraday', 'Keras Prediction', 'Prophet Prediction', 'Sentiment', 'Media', 'ESG', 'DataSeer')
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
            'Payout Ratio': info['payoutRatio'],
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
    
    coMA1, coMA2 = st.columns(2)
    
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
    coBoll1, coBoll2 = st.columns(2)
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
    st.title('Intradays overview')
   
    
    st.subheader('Intraday with interval')
    inter=st.selectbox("Choose the interval",["1m","2m","5m","15m","30m","60m","90m","1d"])
    d1 = st.date_input(
     "Choose the beginning date",
     datetime.date.today())
    st.write('From:', d1)
    d2 = st.date_input(
     "Choose the end date",
     datetime.date.today())
    st.write('To:', d2)

    stock = yf.Ticker(ticker)

    
    Interaday = stock.history(interval=inter,start=d1,end=d2)
    st.table(Interaday)
    def convert_df(df):
        return df.to_csv().encode('utf-8')


    csv = convert_df(Interaday)
    st.download_button(
   "Press to Download",
   csv,
   "file.csv",
   "text/csv",
   key='download-csv'
    )
    
    st.subheader('Intraday per dates')
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
   key='download2-csv'
    )

if(infoType == 'Keras Prediction'):
    d11 = st.date_input(
     "Choose the beginning date of training set",
     datetime.date(1999, 1, 1))
    st.write('From:', d11)
    d22 = st.date_input(
     "Choose the end date of training set",
     datetime.date.today())
    st.write('To:', d22)

    st.title("stock prediction")
    user_input=ticker
    df=pdr.DataReader(ticker, data_source="yahoo",start=d11,end=d22)

    st.subheader("data from 1999")
    st.write(df.describe())

    st.subheader("CLOSING PRICE VS TIME CHART")
    fig=plt.figure(figsize=(12,6))
    plt.plot(df.Close)
    st.pyplot(fig)

    st.subheader("CLOSING PRICE VS TIME CHART & 100MA")
    ma100=df.Close.rolling(100).mean()
    fig=plt.figure(figsize=(12,6))
    plt.plot(ma100, label='Mean 100')
    plt.plot(df.Close)
    plt.legend()
    st.pyplot(fig)

    st.subheader("CLOSING PRICE VS TIME CHART & 100MA and 200MA")
    ma100=df.Close.rolling(100).mean()
    ma200=df.Close.rolling(200).mean()
    fig=plt.figure(figsize=(12,6))
    plt.plot(ma100, label='Mean 100')
    plt.plot(ma200,label='Mean 200')
    plt.plot(df.Close)
    plt.legend()
    st.pyplot(fig)


    #splittng data into data frame
    data_train=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
    data_test=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

    
    scaler=MinMaxScaler(feature_range=(0,1))

    data_train_array=scaler.fit_transform(data_train)

    x_train=[]
    y_train=[]

    #splitting data into xtrain and ytrain
    for i in range(100, data_train_array.shape[0]):
        x_train.append(data_train_array[i-100:i])
        y_train.append(data_train_array[i,0])
    x_train=np.array(x_train)
    y_train=np.array(y_train)

    #load model
    model=load_model('keras_model.h5')
    past_100_days=data_train.tail(100)
    final_df=past_100_days.append(data_test, ignore_index=True)
    input_data=scaler.fit_transform(final_df)

    #testing
    x_test=[]
    y_test=[]

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i,0])
    x_test,y_test= np.array(x_test),np.array(y_test)
    y_predicted=model.predict(x_test)

    scaler=scaler.scale_
    scaler_factor=1/scaler[0]
    y_predicted=y_predicted*scaler_factor
    y_test=y_test*scaler_factor

    # showing
    st.subheader("PREDICTION VS TIME ORIGINAL PRICE")
    fig2=plt.figure(figsize=(12,6))
    plt.plot(y_test,'b', label='Original Price')
    plt.plot(y_predicted,'r',label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    config={
        'modeBarButtonsToAdd': ['drawline']
    }
    st.plotly_chart(fig2, config=config)
    st.subheader("Comparing last five predicted and real prices")
    predf1=y_test[-5:]
    predf2=y_predicted[-5:]

if(infoType == 'Prophet Prediction'):
   
   tickerSymbol = ticker
   tickerData = yf.Ticker(tickerSymbol)
   alpha2,gamma2 = st.columns(2)
   starting = alpha2.date_input('Pick a starting Date')
   ending = gamma2.date_input('Pick an ending date')


   rhs, rh, rhs1 = st.columns((2.5,3,2))
   rh.markdown("""### Let's have a look at some raw data""")
   #period = st.radio('Choose the Duration of Schemantics',('1d','1mo','1y'))

   def load_data(ticker):
       data = yf.download(ticker, starting, ending)
       data.reset_index(inplace=True)
       return data
   data = load_data(tickerSymbol)
   rds, rd, rds1 = st.columns((1,5,1))
   rd.write(data)

   tickerDF=tickerData.history(period='1d', start=starting, end=ending)


   graphs1, graphs3 = st.columns(2)
   graphs1.markdown("""### Opening Price """)
   graphs1.line_chart(tickerDF.Open)
   graphs3.markdown("""### Volume Price """)
   graphs3.line_chart(tickerDF.Volume)
   graphs1.markdown("""### Closing Price """)
   graphs1.line_chart(tickerDF.Close)
   graphs3.markdown("""### Highest Price """)
   graphs3.line_chart(tickerDF.High)
   graphs1.markdown("""### Lowest Price""")
   graphs1.line_chart(tickerDF.Low)
   graphs3.markdown("""### Adjusted Closing Price """)
   graphs3.line_chart(data['Adj Close'])


   START = "2015-01-01"
   TODAY = date.today().strftime("%Y-%m-%d")

   @st.cache
   def load_data1(ticker):
       data1 = yf.download(ticker, START, TODAY)
       data1.reset_index(inplace=True)
       return data1

   data1 = load_data1(tickerSymbol)
   #st.write(data1)

   df_train = data1[['Date','Close']]
   df_train = df_train.rename(columns = {'Date':'ds','Close':'y'})
   #st.write(df_train)

   predictor = Prophet()
   predictor.fit(df_train)

   st.markdown("""## Please Select the number of years for the predicted trends""")

   n_years = st.slider('Years of prediction:', 1, 4)
   period = n_years * 365
   future = predictor.make_future_dataframe(periods = period)
   forecast = predictor.predict(future)

   fcs2 , afc, fcs3 = st.columns(3)
   # Show and plot forecast
   afc.title('Forecast data')
   #st.write(forecast.tail())
    
   afc.markdown(f'Forecast plot for {n_years} following years')
   fig1 = plot_plotly(predictor, forecast)
   st.plotly_chart(fig1)


   fcs,fch,fcs1 = st.columns(3)
   fch.markdown("""## Forecated Components""")
   fig2 = predictor.plot_components(forecast)
   st.write(fig2)

   st.caption('Made with Rivaldi')	
	

if(infoType == 'Sentiment'):
    
    def get_news(ticker):
        url = finviz_url + ticker
        req = Request(url=url,headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0'}) 
        response = urlopen(req)    
        # Read the contents of the file into 'html'
        html = BeautifulSoup(response)
        # Find 'news-table' in the Soup and load it into 'news_table'
        news_table = html.find(id='news-table')
        return news_table
        # parse news into dataframe
    def parse_news(news_table):
        parsed_news = []
    
        for x in news_table.findAll('tr'):
            # read the text from each tr tag into text
            # get text from a only
            text = x.a.get_text() 
            # splite text in the td tag into a list 
            date_scrape = x.td.text.split()
            # if the length of 'date_scrape' is 1, load 'time' as the only element

            if len(date_scrape) == 1:
                time = date_scrape[0]
            
            # else load 'date' as the 1st element and 'time' as the second    
            else:
                date = date_scrape[0]
                time = date_scrape[1]
        
            # Append ticker, date, time and headline as a list to the 'parsed_news' list
            parsed_news.append([date, time, text])
        
            # Set column names
            columns = ['date', 'time', 'headline']

            # Convert the parsed_news list into a DataFrame called 'parsed_and_scored_news'
            parsed_news_df = pd.DataFrame(parsed_news, columns=columns)
        
            # Create a pandas datetime object from the strings in 'date' and 'time' column
            parsed_news_df['datetime'] = pd.to_datetime(parsed_news_df['date'] + ' ' + parsed_news_df['time'])
        
        return parsed_news_df
        

	

    def score_news(parsed_news_df):
        # Instantiate the sentiment intensity analyzer
        vader = SentimentIntensityAnalyzer()
    
        # Iterate through the headlines and get the polarity scores using vader
        scores = parsed_news_df['headline'].apply(vader.polarity_scores).tolist()

        # Convert the 'scores' list of dicts into a DataFrame
        scores_df = pd.DataFrame(scores)

        # Join the DataFrames of the news and the list of dicts
        parsed_and_scored_news = parsed_news_df.join(scores_df, rsuffix='_right')
    
            
        parsed_and_scored_news = parsed_and_scored_news.set_index('datetime')
    
        parsed_and_scored_news = parsed_and_scored_news.drop(['date', 'time'], 1)    
        
        parsed_and_scored_news = parsed_and_scored_news.rename(columns={"compound": "sentiment_score"})

        return parsed_and_scored_news

    def plot_hourly_sentiment(parsed_and_scored_news, ticker):
   
        # Group by date and ticker columns from scored_news and calculate the mean
        mean_scores = parsed_and_scored_news.resample('H').mean()

        # Plot a bar chart with plotly
        fig = px.bar(mean_scores, x=mean_scores.index, y='sentiment_score', title = ticker + ' Hourly Sentiment Scores')
        return fig # instead of using fig.show(), we return fig and turn it into a graphjson object for displaying in web page later

    def plot_daily_sentiment(parsed_and_scored_news, ticker):
   
        # Group by date and ticker columns from scored_news and calculate the mean
        mean_scores = parsed_and_scored_news.resample('D').mean()

        # Plot a bar chart with plotly
        fig = px.bar(mean_scores, x=mean_scores.index, y='sentiment_score', title = ticker + ' Daily Sentiment Scores')
        return fig # instead of using fig.show(), we return fig and turn it into a graphjson object for displaying in web page later
    def convert_df(df):
        return df.to_csv().encode('utf-8')

    st.header("Stocks Sentiment based on news from FinViz")
    st.subheader("Hourly and Daily Sentiment of {} Stock".format(ticker))
    description = '''
    The news headlines for the sentiment score are obtained from the FinViz website.
    The following charts averages the sentiment scores of {} stock hourly and daily.
    While the table below gives  the most recent headlines of the stock with the negative, neutral, positive and an aggregated sentiment score.
    '''.format(ticker)
    news_table = get_news(ticker)
    parsed_news_df = parse_news(news_table)
    parsed_and_scored_news = score_news(parsed_news_df)
    fig_hourly = plot_hourly_sentiment(parsed_and_scored_news, ticker)
    fig_daily = plot_daily_sentiment(parsed_and_scored_news, ticker)
    st.plotly_chart(fig_hourly)
    st.plotly_chart(fig_daily)

 
    st.write(description)
    st.table(parsed_and_scored_news)
    csv = convert_df(parsed_and_scored_news)
    st.download_button(
   "Press to Download",
   csv,
   "file.csv",
   "text/csv",
   key='download-csv'
    )
if(infoType == 'ESG'):

   esg_data = pd.DataFrame()
   stock_df = stock.sustainability.T
   stock_df['symbol'] = ticker
   esg_data = esg_data.append(stock_df)
   new_esg_df = esg_data[['symbol', 'peerGroup', 'socialScore', 'governanceScore', 'totalEsg', 'environmentScore', 'esgPerformance']]

   def convert_df(df):
        return df.to_csv().encode('utf-8')
   st.header("ESG Stock Analysis")
   Description2 = '''
       The ESG score is obtained from the Yahoo Finance website.
       The following chart gives the overall three ESG score of {} stock
       While the table gives  the ESG Performance given from the website.
       '''.format(ticker)
   st.write(Description2)
   st.subheader("ESG Score Table")
   st.table(new_esg_df)
   csv = convert_df(new_esg_df)
   st.download_button(
   "Press to Download",
   csv,
   "file.csv",
   "text/csv",
   key='download-csv'
    )
   st.subheader("ESG Radar")

   
   df_esg = pd.DataFrame(dict(
       r=[new_esg_df.iloc[0,2], new_esg_df.iloc[0,3], new_esg_df.iloc[0,5]],
       theta=['Social Score','Governance Score','Environment Score',]))
   fig = px.line_polar(df_esg, r='r', theta='theta', line_close=True)
   st.plotly_chart(fig)

if(infoType == 'Media'):
	
   st.title('Under Construction')
   #def displayPDF(file):
	# Opening file from file path
        #with open(file, "rb") as f:
            #base64_pdf = base64.b64encode(f.read()).decode('utf-8')

        # Embedding PDF in HTML
        #pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'

        # Displaying File
        #return st.markdown(pdf_display, unsafe_allow_html=True)
   
   #displayPDF("Media/Twitter_ADOBE.pdf")
   p = open("Media/Impact_period2.html", 'r')
   components.iframe(p.read())
	
if(infoType == 'DataSeer'):
	
   st.title('Under Construction')	
