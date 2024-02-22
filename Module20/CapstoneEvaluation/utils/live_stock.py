
from typing import Any
from alpaca.data.live.stock import StockDataStream


from decouple import config

api_key = config("ALPACA_KEY")
secret_key = config("ALPACA_SECRET")
live_data_url = config("ALPACA_WSS_URL")


wss_client = StockDataStream(api_key=api_key, secret_key=secret_key) #, url_override=live_data_url)

# async handler
async def quote_data_handler(data: Any):
    # quote data will arrive here
    print(data)

#wss_client.subscribe_quotes(quote_data_handler, "BTC/USD")
wss_client.subscribe_bars(quote_data_handler, "JPM")

wss_client.run()
