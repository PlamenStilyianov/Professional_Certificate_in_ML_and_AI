import warnings
import numpy as np
import pandas as pd
from finta import TA as finta
import matplotlib.pyplot as plt
from plotly.figure_factory import create_table
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score

warnings.filterwarnings("ignore")

COLUMNS = ['open', 'high', 'low', 'close', 'volume']


def add_pct_changes(df_stock):
    columns = ['open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']
    df = df_stock[columns]
    df['HL_PCT'] = (df['high'] - df['low']) / df['close'] * 100.0
    df['PCT_change'] = (df['close'] - df['open']) / df['open'] * 100.0
    return df


# def add_indicators_pd_ta(df_stock):
#     indicators = ['KAMA', 'RSI', 'DEMA', 'PPO', 'SMA', 'WMA', 'MOM', 'ROC', 'TRIX', 'TRIMA', 'EMA', 'TEMA', 'MACD']
#     df_ta = df_stock['vwap']
#     for indicator in indicators:
#         df = None
#         try:
#             df = eval('ta.' + str(indicator).lower() + '(df_ta)')
#         except:
#             print('Invalid Indicator: ' + indicator)
#
#         if isinstance(df, pd.Series):
#             df = df.to_frame()
#         # Appropriate labels on each column
#         df = df.add_prefix(indicator + '_')
#         # Join merge dataframes based on the date
#         df_stock = df_stock.merge(df, left_index=True, right_index=True)
#
#     df = None
#     df = eval('ta.' + str('bbands') + '(df_ta)')
#     df_stock = df_stock.merge(df, left_index=True, right_index=True)
#     # Fix labels
#     df_stock.columns = df_stock.columns.str.replace(' ', '_')
#     df_stock.index.name = 'date'
#     return df_stock
#
#
# def add_indicators_talib(df_stock):
#     indicators = ['KAMA', 'RSI', 'DEMA', 'PPO', 'SMA', 'WMA', 'MOM', 'ROC', 'TRIX', 'TRIMA', 'EMA', 'TEMA', 'MACD']
#     df_talib = df_stock['close']
#     for indicator in indicators:
#         df = None
#         # Using python's eval function to create a method from a string instead of having every method defined
#         try:
#             if indicator == 'MACD':
#                 df = eval('talib.' + indicator + '(df_talib)')
#             else:
#                 df = eval('talib.' + indicator + '(df_talib, 2)')
#         except:
#             print(indicator)
#             continue
#         # Some method return series, so we can check to convert here
#         if isinstance(df, pd.Series):
#             df = df.to_frame()
#         else:
#             for s in df:
#                 df_t = s.to_frame()
#                 df_t = df_t.add_prefix(indicator + '_')
#                 # Join merge dataframes based on the date
#                 df_stock = df_stock.merge(df_t, left_index=True, right_index=True)
#             continue
#         # Appropriate labels on each column
#         df = df.add_prefix(indicator + '_')
#         # Join merge dataframes based on the date
#         df_stock = df_stock.merge(df, left_index=True, right_index=True)
#     # Fix labels
#     df_stock.columns = df_stock.columns.str.replace(' ', '_')
#     df_stock.index.name = 'date'
#     return df_stock
#
#
# def add_indicators_ta(df_stock):
#     df = tta.add_all_ta_features(df_stock, *COLUMNS)
#     df_stock = df_stock.merge(df, left_index=True, right_index=True)
#     # Fix labels
#     df_stock.columns = df_stock.columns.str.replace(' ', '_')
#     df_stock.index.name = 'date'
#     return df_stock


def add_indicators_finta(df_stock):
    # Here are all indicators we are using
    indicators = ['SMM', 'SSMA', 'EMA', 'DEMA', 'TEMA', 'TRIX', 'VAMA', 'ER', 'ZLEMA', 'WMA',
                  'HMA', 'EVWMA', 'VWAP', 'SMMA', 'MACD', 'PPO', 'VW_MACD', 'EV_MACD', 'MOM', 'ROC', 'RSI', 'IFT_RSI']
    # indicators = [name for name in list(fta.__dict__.keys()) if str(name).isupper()]
    # These indicators need more tuning or are broken
    broken_indicators = ['SAR', 'TMF', 'VR', 'QSTICK']

    df_fa = df_stock[COLUMNS]
    for indicator in indicators:
        if indicator not in broken_indicators:
            df = None
            # Using python's eval function to create a method from a string instead of having every method defined
            try:
                df = eval('finta.' + indicator + '(df_fa)')
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
    df_stock.index.name = 'date'
    return df_stock


def get_news_scores(df):
    scores = {'date': [], 'score': [], 'news_type': []}
    for index in set(df.index):
        if isinstance(df.loc[index]['score'], pd.core.series.Series):
            scores['date'].append(index)
            n_type = df.loc[index]['news_type'].max()
            n_df = df.loc[index][['score', 'news_type']].query("news_type == @n_type")
            scores['score'].append(np.mean(n_df['score']))
            scores['news_type'].append(n_type)
        else:
            scores['date'].append(index)
            scores['score'].append(df.loc[index]['score'])
            scores['news_type'].append(df.loc[index]['news_type'])

    news = pd.DataFrame.from_dict(scores)
    news.set_index('date', inplace=True)
    news.sort_index(ascending=True, inplace=True)
    news.index.name = 'date'
    return news


def trim_df_columns(df_stocks_full):
    for column in df_stocks_full.columns.to_list():
        if df_stocks_full[column].isna().sum() > 0 or column in ['SQZMI_20_period_SQZMI']:
            df_stocks_full.drop(columns=[column], axis=1, inplace=True)

    return df_stocks_full


def print_permutation_importance(models, X_val, y_val, df, X):
    for model, name in models:
        print(f'Permutation importance to Close for {model[name].__class__.__name__ if name != "clf" else model.__class__.__name__}')
        r = permutation_importance(model, X_val, y_val, n_repeats=32, random_state=0)
        for i in r.importances_mean.argsort()[::-1]:
            #if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f"{X.columns[i]:<{df.shape[1]}}"
                  f"{r.importances_mean[i]:.3f}"
                  f" +/- {r.importances_std[i]:.3f}")
        print(' ')

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

        # 1 & 2 Coefficient of Determination (R² & Adjusted R²)
        r2 = r2_score(y_test, y_pred)
        adj_r2 = 1 - (1 - r2) * (len(y_train) - 1) / (len(y_train) - X_train.shape[1] - 1)
        R2_result.append(round(r2, 2))
        adj_R2_result.append(round(adj_r2, 2))

        # 3 & 4. MSE and RMSE
        mse = mean_squared_error(y_pred=y_pred, y_true=y_test, squared=True)
        rmse = mean_squared_error(y_pred=y_pred, y_true=y_test, squared=False)
        MSE_result.append(round(mse, 2))
        RMSE_result.append(round(rmse, 2))

        # 5. MAE
        mae = mean_absolute_error(y_pred=y_pred, y_true=y_test)
        MAE_result.append(round(mae, 2))

        # 6. Model training and test scores or accuracies
        train_score = round(pred_model.score(X_train, y_train) * 100, 2)
        test_score = round(pred_model.score(X_test, y_test) * 100, 2)

        training_score.append(train_score)
        testing_score.append(test_score)

        number_of_observations = 60
        index_point = X_test.shape[0] - number_of_observations
        x_ax = X_test.index.tolist()[index_point:]

        #    vwap = X_test['vwap']

        plt.figure(figsize=(25, 10))

        plt.plot(x_ax, y_test[index_point:], label="Actual", color='red', linewidth=2)
        plt.plot(x_ax, y_pred[index_point:], label="Predicted", color='yellow', linewidth=2)
        #   plt.plot(x_ax, vwap[index_point:], label="VWAP", color='blue', linewidth=2)

        plt.title("JPM Close Prices: Predicted data - Actual using " + model_name)
        plt.xlabel('Years')
        plt.ylabel('Price')
        start = min(x_ax)
        end = max(x_ax)
        t = pd.date_range(start=start, end=end, freq='M')
        plt.xticks(t)
        plt.gcf().autofmt_xdate()
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid()
        plt.show();

    pd_metric = pd.DataFrame(
        {'models': str_models, 'R2': R2_result, 'Adj_R2': adj_R2_result, 'MSE': MSE_result, 'RMSE': RMSE_result,
         'MAE': MAE_result, 'Training_Score': training_score, 'Test_Score': testing_score})
    pd_metric.set_index('models', inplace=True)
    return create_table(pd_metric, index_title='Models', index=True)


def get_complexity(X_train, X_test, y_train, y_test, complex_num):
    train_mses = []
    test_mses = []
    r_squared = []

    for i in range(1, complex_num + 1):
        poly_ordinal_ohe = make_column_transformer(
            (PolynomialFeatures(include_bias=False, degree=i), make_column_selector(dtype_include=np.number)))
        pipe = Pipeline([('transformer', poly_ordinal_ohe), ('linreg', LinearRegression())])

        # fit on train
        pipe.fit(X_train, y_train)
        R_squared = r2_score(y_test, pipe.predict(X_test))

        # predict on train and test
        p1 = pipe.predict(X_train)
        p2 = pipe.predict(X_test)

        # create MSEs for train and test sets
        train_mses.append(round(mean_squared_error(y_train, p1), 4))
        test_mses.append(round(mean_squared_error(y_test, p2), 4))
        r_squared.append(round(R_squared, 4))
    best_complexity = test_mses.index(min(test_mses)) + 1
    best_mse = min(test_mses)
    best_rsq2 = max(r_squared)

    # Answer check
    print(f'The best degree of the polynomial model is:  {best_complexity} out of {complex_num}')
    print(f'The smallest mean squared error on the test dataset is : {best_mse: .4f} out of {test_mses}')
    print(f'The best value of the R-sq of the model, as a good fit is: {best_rsq2: .4f} out of {r_squared}')
    return best_complexity

#%%
