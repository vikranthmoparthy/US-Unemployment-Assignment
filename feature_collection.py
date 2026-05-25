import pandas as pd
from fredapi import Fred
import yfinance as yf
from pytrends.request import TrendReq
import time
import requests

fred = Fred(api_key='b17c6f7340c4d2de0fbc4a4ab463cd9a')
pytrends = TrendReq(hl='en-US', tz=360)

START_DATE = '2000-01-01'
END_DATE = pd.Timestamp.today().strftime('%Y-%m-%d')

fred_series = {
    'UNRATE': 'Unemployment_Rate',
    'ICSA': 'Initial_Jobless_Claims',       
    'PAYEMS': 'Nonfarm_Payrolls',
    'FEDFUNDS': 'Fed_Funds_Rate',
    'T10Y2Y': 'Yield_Curve_Spread',         
    'CPIAUCSL': 'CPI_Headline',
    'UMCSENT': 'Consumer_Sentiment',        
    'RSAFS': 'Retail_Sales',
    'INDPRO': 'Industrial_Production',
    'HOUST': 'Housing_Starts',
    'VIXCLS': 'VIX',                        
    'BAA10Y': 'Credit_Spread'               
}

search_terms = ["layoffs", "unemployment benefits", "jobs near me"]

macro_data = pd.DataFrame()

for ticker, name in fred_series.items():
    try:
        series = fred.get_series(ticker, observation_start=START_DATE)
        df_temp = pd.DataFrame(series, columns=[name])
        df_temp = df_temp.resample('ME').mean()
        
        if macro_data.empty:
            macro_data = df_temp
        else:
            macro_data = macro_data.join(df_temp, how='outer')
        time.sleep(0.2)
    except Exception as e:
        pass

try:
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    
    sp500 = yf.download('^GSPC', start=START_DATE, end=END_DATE, progress=False, session=session)
    
    if not sp500.empty:
        sp500_close = sp500[['Close']].copy()
        sp500_close.columns = ['SP500_Close']
        sp500_monthly = sp500_close.resample('ME').mean()
        macro_data = macro_data.join(sp500_monthly, how='outer')
        
except Exception as e:
    pass

pytrends_robust = TrendReq(
    hl='en-US', 
    tz=360, 
    requests_args={'headers': {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0'}}
)

max_retries = 3
for attempt in range(max_retries):
    try:
        pytrends_robust.build_payload(search_terms, cat=0, timeframe=f'2004-01-01 {END_DATE}', geo='US', gprop='')
        trends_data = pytrends_robust.interest_over_time()
        
        if not trends_data.empty:
            trends_data = trends_data.drop(columns=['isPartial'])
            trends_data.columns = [f"Trend_{term.replace(' ', '_')}" for term in trends_data.columns]
            trends_monthly = trends_data.resample('ME').mean()
            macro_data = macro_data.join(trends_monthly, how='outer')
            break
        else:
            break
            
    except Exception as e:
        if attempt < max_retries - 1:
            sleep_time = 10 * (attempt + 1)
            time.sleep(sleep_time)

macro_data.dropna(how='all', inplace=True)
macro_data.ffill(inplace=True)
macro_data = macro_data.loc[START_DATE:]
macro_data.index = macro_data.index.strftime('%Y-%m-%d')
macro_data.index.name = 'Date'

csv_filename = 'us_macro_features_extended.csv'
macro_data.to_csv(csv_filename)