import datetime
import random

import matplotlib.pyplot as plt
import numpy as np
import quandl
from matplotlib import style

style.use('ggplot')

df = quandl.get("WIKI/GOOGL")
df = df[['Adj. Close']]

df.dropna(inplace=True)

# Prepare variables for loop

last_close = df['Adj. Close'].iloc[-1]
last_date = df.iloc[-1].name.timestamp()
df['Forecast'] = np.nan

for i in range(1000):
    # Create np.Array of current predictions to serve as input for future predictions
    modifier = random.randint(-100, 105) / 10000 + 1
    last_close *= modifier
    next_date = datetime.datetime.fromtimestamp(last_date)
    last_date += 86400

    # Outputs data into DataFrame to enable plotting
    df.loc[next_date] = [np.nan, last_close]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
