## Predicting the Stock Trend with News Sentiment Analysis and Technical Indicators 

**Plamen Stilyianov**

### Executive summary

**Project overview and goals:** The goal of this project is to predict the Stock Trend with News Sentiment Analysis and Technical Indicators.

Among all the different features considered useful in stock prediction as outlined by Tatsat et al, (2020), technical indicators 
and news are the most popular and representative: technical indicators may identify patterns behind the stock price times series, 
while financial news can indicate upcoming events that influence investors’ decisions. The use of the machine learning classification and regression frameworks 
to build a future stock trend prediction model that combines technical analysis and fundamental analysis (financial news) and reported an excellent backtest performance.

A six-step methodology has been used as: data collection, data preprocessing, feature generation, sentiment analysis, feature correlation analysis, and applying machine learning algorithms.

**Findings:** The best model for predicting the stock trend RandomForestRegressor model, which 
has the lowest Root Mean Square Error (RMSE) 19.56, It shows how far predictions fall from measured true values using Euclidean distance.
A R-squared between 0.50 to 0.99 is acceptable in social science research especially when most of the explanatory variables are statistically significant.
<img src="images/reg_table.png">

**Results and conclusion:** At the moment of developing the RandomForestRegressor model is the best performer, but next development is the Long Short-Term Memory Networks model.
The Long Short-Term Memory Networks is a deep learning, sequential neural network that allows information to persist. It is a special type of Recurrent Neural Network which is capable of handling the vanishing gradient problem faced by RNN. LSTM was designed by Hochreiter and Schmidhuber that resolves the problem caused by traditional rnns and machine learning algorithms. LSTM Model can be implemented in Python using the Keras library.
The LSTM model is then trained to predict the target variable (e.g., the future stock price) based on the historical information.

**Data Collection**
The data consists of two parts, the financial data, and the financial news data. The former is the standard Alpaca historical data of the mentioned companies from 2016 to 2024.

**Data Preprocessing**
A simple preprocessing was performed on the raw news data. They kept only date and article columns, filtered the news relevant to the three companies by keyword search, and converted the raw texts into a cleaner format.

**Feature Generation**
Next, some additional features and three popular technical indicators were created out of the stock price information: Today Trend, Tomorrow Trend, RSI (Relative Strength Index), SMA (Simple Moving Average),

**Sentiment Analysis**
Transformers are used as a sentiment analysis tool to analyze the sentiment of each news article. Having obtained a sentiment score for each piece of news, they further grouped them by date; the overall sentiment score of any day is the aggregate score of all news on that date. “When the overall sentiment is high on a given day, the stock tends to receive much attention from traders and investors”.

**Feature Correlation Analysis**
If we look at the pairwise correlation matrix, the price features and indicator features are highly correlated, it was kept only the low correlated and removed the rest in the final dataset

**Data Collection and Preprocessing**
The OHLCV historical data of JP Morgan (JPM) over the two years from 2016 to 2024 can be easily loaded with the help of Alpaca Markets. A preliminary visualization of the closing price movement tells us how volatile this stock was in the seven years.
<img src="images/jpm_close.png">

The relevant news headlines have been well organized in the previously mentioned dataset; let’s load it and take a glance.
<img src="images/jpm_news.png">

