import pandas as pd
import math as math
from datetime import datetime
from alpaca.data import StockHistoricalDataClient, TimeFrame
from alpaca.data.requests import StockBarsRequest
from decouple import config

api_key = config("ALPACA_KEY")
secret_key = config("ALPACA_SECRET")

# Instantiate a data client
data_client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)

day = datetime.now().day if math.log10(datetime.now().day) >= 1 else f'0{datetime.now().day}'
month = datetime.now().month if math.log10(datetime.now().month) >= 1 else f'0{datetime.now().month}'
year = datetime.now().year

# Set the start time
start_time = pd.to_datetime("2015-01-15").tz_localize('America/New_York')
end_time = datetime.fromisoformat(f'{year}-{month}-{day}T00:00:00Z')

# It's generally best to explicitly provide an end time but will default to 'now' if not
request_params = StockBarsRequest(
    symbol_or_symbols=['JPM'],
    timeframe=TimeFrame.Day,
    start=start_time,
    end=end_time
    )

bars_df = data_client.get_stock_bars(request_params).df.tz_convert('America/New_York', level=1)
bars_df.to_csv(f'../data/jpm_bars_{year}-{month}-{day}.csv', index=True)
print(bars_df.head())
print(bars_df.tail())