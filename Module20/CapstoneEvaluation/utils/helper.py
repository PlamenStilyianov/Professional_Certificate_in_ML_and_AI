import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from finta import TA
from plotly.figure_factory import create_table
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

def add_indicators(df_stock):

    # Here are all indicators we are using
    #indicators = ['SMA', 'SMM', 'SSMA', 'EMA', 'DEMA', 'TEMA', 'TRIMA', 'TRIX', 'VAMA', 'ER', 'KAMA', 'ZLEMA', 'WMA', 'HMA', 'EVWMA', 'VWAP', 'SMMA', 'MACD', 'PPO', 'VW_MACD', 'EV_MACD', 'MOM', 'ROC', 'RSI', 'IFT_RSI']
    indicators = [name for name in list(TA.__dict__.keys()) if str(name).isupper()]
    # These indicators need more tuning or are broken
    broken_indicators = ['SAR', 'TMF', 'VR', 'QSTICK']
    columns = ['open', 'high', 'low', 'close', 'volume']
    df_fa = df_stock[columns]
    for indicator in indicators:
        if indicator not in broken_indicators:
            df = None
            # Using python's eval function to create a method from a string instead of having every method defined
            try:
                df = eval('TA.' + indicator + '(df_fa)')
            except:
                print(indicator)
                continue
            # Some method return series, so we can check to convert here
            if not isinstance(df, pd.DataFrame):
                df = df.to_frame()
            # Appropriate labels on each column
            df = df.add_prefix(indicator + '_')
            # Join merge dataframes based on the date
            df_stock = df_stock.merge(df, left_index=True, right_index=True)
    # Fix labels
    df_stock.columns = df_stock.columns.str.replace(' ', '_')
    return df_stock


def get_news_scores(df):
    scores = {'date':[], 'score':[], 'news_type':[]}
    for index in set(df.index):
        if isinstance(df.loc[index]['score'], pd.core.series.Series):
            scores['date'].append(index)
            n_type = df.loc[index]['news_type'].max()
            n_df = df.loc[index][['score', 'news_type']].query("news_type == @n_type")
            scores['score'].append( np.mean(n_df['score']) )
            scores['news_type'].append(n_type)
        else:
            scores['date'].append(index)
            scores['score'].append(df.loc[index]['score'])
            scores['news_type'].append(df.loc[index]['news_type'])

    news = pd.DataFrame.from_dict(scores)
    news.set_index('date', inplace=True)
    news.sort_index(ascending=True, inplace=True)

    return news

def trim_df_columns(df_stocks_full):
    for column in df_stocks_full.columns.to_list():
        if df_stocks_full[column].isna().sum() > 0 or column in ['SQZMI_20_period_SQZMI']:
            df_stocks_full.drop(columns=[column], axis=1, inplace=True)

    return df_stocks_full


# Create an ensamble method to select best fit model
def reg_model_metrics(reg_models, X_train, X_test, y_train, y_test):
    """ Function takes in different dictionary of models and training and testing sets,
    and ouputs the below metrics in a dataframe:
    1. R² or Coefficient of Determination.
    2. Adjusted R²
    3. Mean Squared Error(MSE)
    4. Root-Mean-Squared-Error(RMSE).
    5. Mean-Absolute-Error(MAE).
    6. Model training and test scores or accuracies
    7. Plots graph between actual vs predicted datasets
    """

    R2_result = []
    adj_R2_result = []
    MSE_result = []
    RMSE_result = []
    MAE_result = []
    str_models = []
    training_score = []
    testing_score = []

    for model_name, model in reg_models.items():
        # Get predicted values on x_test
        pred_model = model.fit(X_train, y_train)
        y_pred = pred_model.predict(X_test)
        str_models.append(str(model_name))

        #1 & 2 Coefficient of Determination (R² & Adjusted R²)
        r2 = r2_score(y_test, y_pred)
        adj_r2 = 1 - (1 - r2) * (len(y_train) - 1) / (len(y_train) - X_train.shape[1] - 1)
        R2_result.append(round(r2, 2))
        adj_R2_result.append(round(adj_r2, 2))

        #3 & 4. MSE and RMSE
        mse = mean_squared_error(y_pred=y_pred, y_true=y_test, squared=True)
        rmse = mean_squared_error(y_pred=y_pred, y_true=y_test, squared=False)
        MSE_result.append(round(mse, 2))
        RMSE_result.append(round(rmse, 2))

        #5. MAE
        mae = mean_absolute_error(y_pred=y_pred, y_true=y_test)
        MAE_result.append(round(mae, 2))

        #6. Model training and test scores or accuracies
        train_score = round(pred_model.score(X_train, y_train) * 100, 2)
        test_score = round(pred_model.score(X_test, y_test) * 100, 2)

        training_score.append(train_score)
        testing_score.append(test_score)

        number_of_observations = 50
        index_point = X_test.shape[0] - number_of_observations
        x_ax = X_test.index.tolist()[index_point:]

        vwap = X_test['vwap']

        plt.figure(figsize=(25, 10))

        plt.plot(x_ax, y_test[index_point:], label="Actual", color='red', linewidth=2)
        plt.plot(x_ax, y_pred[index_point:], label="Predicted", color='yellow', linewidth=2)
        plt.plot(x_ax, vwap[index_point:], label="VWAP", color='blue', linewidth=2)

        plt.title("JPM Close Prices: Predicted data - Actual using " + model_name)
        plt.xlabel('Years')
        plt.ylabel('Price')
        plt.xticks(X_test.index.tolist()[index_point:])
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid()
        plt.show();

    pd_metric = pd.DataFrame(
        {'models': str_models, 'R2': R2_result, 'Adj_R2': adj_R2_result, 'MSE': MSE_result, 'RMSE': RMSE_result,
         'MAE': MAE_result, 'Training_Score': training_score, 'Test_Score': testing_score})
    pd_metric.set_index('models', inplace=True)
    return create_table(pd_metric, index_title='Models', index=True)