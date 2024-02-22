import warnings
import pandas as pd
import math
from datetime import datetime
from transformers import pipeline
from decouple import config
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest

warnings.filterwarnings("ignore")

api_key = config("ALPACA_KEY")
secret_key = config("ALPACA_SECRET")
news_url = config("ALPACA_DATA_URL")

client = NewsClient(api_key, secret_key, url_override=news_url)

classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

day = datetime.now().day if math.log10(datetime.now().day) >= 1 else f'0{datetime.now().day}'
month = datetime.now().month if math.log10(datetime.now().month) >= 1 else f'0{datetime.now().month}'
year = datetime.now().year

start = datetime.fromisoformat('2015-01-01T00:00:00Z')
end = datetime.fromisoformat(f'{year}-{month}-{day}T00:00:00Z')
news_entries = {'created_at': [], 'updated_at': [], 'headline': [], 'summary': [], 'url':[], 'news_type': [], 'score': [], 'symbols': []}
df_news = pd.DataFrame(news_entries, columns=['created_at', 'updated_at', 'headline', 'summary', 'url', 'news_type', 'score', 'symbols'])

page_token = 'default'
while page_token is not None:

    request_params = NewsRequest(
        start=start.isoformat(),
        end=end.isoformat(),
        symbols='JPM',
        limit=50,
        page_token=page_token if page_token!='default' else None
    )

    respond = client.get_news(request_params)

    news_data = respond.news
    for news in news_data:
        created_at = news.created_at.isoformat().split('+')[0].replace('T',' ')
        headline = news.headline
        url = news.url
        summary = news.summary
        sentiment = classifier(summary + headline)[0]
        score = sentiment['score']
        news_type = sentiment['label']
        updated_at = news.updated_at.isoformat().split('+')[0].replace('T',' ')
        symbols = news.symbols
        entries = {'created_at': created_at, 'updated_at': updated_at, 'headline': headline, 'summary': summary, 'url': url, 'news_type': news_type, 'score': score, 'symbols': symbols }
        row = pd.Series(entries)
        df_news = pd.concat([df_news, row.to_frame().T], axis=0, ignore_index=True)
    page_token = respond.next_page_token

print(df_news.head())
print(df_news.tail())
df_news.to_csv(f'../data/jpm_news_{year}-{month}-{day}.csv')
