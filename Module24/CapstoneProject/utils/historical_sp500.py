import yfinance as yf


def get_historical_sp500(tickers='^GSPC', start_date='', end_date=''):
    df_sp500 = yf.download(tickers, start=start_date, end=end_date)
    df_sp500.reset_index(inplace=True)
    df_sp500.rename(
        columns={'Date': "date", 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'},
        inplace=True)
    df_sp500['adj close'] = df_sp500['Adj Close']
    df_sp500.drop(columns=['Adj Close'], inplace=True)
   # df_sp500.reset_index(inplace=True)
    df_sp500.set_index('date', inplace=True)
  #  df_sp500.drop(columns=['index'], inplace=True)
    print(df_sp500.head(3))
    print(df_sp500.tail(3))
    print(df_sp500.index)
    year, month, day = df_sp500.index[-1].year, df_sp500.index[-1].month, df_sp500.index[-1].day
    df_sp500.to_csv(f'../data/sp500_bars_{year}-{month}-{day}.csv', index=True)


if __name__ == '__main__':
    get_historical_sp500('^GSPC', start_date='2016-01-01', end_date='2024-03-25')

#%%
