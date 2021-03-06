import sklearn
import numpy as np
import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
from functools import reduce
from sklearn.metrics import mean_absolute_error, mean_squared_error
from fbprophet import Prophet
import plotly.graph_objects as go
import argparse
# Load your cloud drive
# from google.colab import drive
# drive.mount('/content/drive')

def run(train,test,output):
  # 讀入series
  colnames=['Open','High','Low','Close'] 
  df_temp = pd.read_csv(train, names=colnames)
  date = pd.date_range(start='2015/01/01',periods=len(df_temp))
  df = df_temp
  date = pd.DataFrame(date)
  df[['Date']] = date
  #df.set_index('Date',inplace=True)
  price = df.squeeze()
  price.head()

  # 讀入series
  colnames=['Open','High','Low','Close'] 
  df_temp = pd.read_csv(test, names=colnames)
  date2 = pd.date_range(start=date.iloc[len(date)-1,0],periods=2)
  date2 = pd.DataFrame(date2)
  date2 = pd.date_range(start=date2.iloc[len(date2)-1,0],periods=len(df_temp))
  date2 = pd.DataFrame(date2)
  df2 = df_temp
  df2[['Date']] = date2
  #df2.set_index('Date',inplace=True)
  price2 = df2.squeeze()
  price2.head()

  df = pd.concat([df,df2])
  df = df.reset_index()
  df = df[['Open','High','Low','Close','Date']]
  df

  def relative_strength_idx(df, n=14):
      Open = df['Open']
      delta = Open.diff()
      delta = delta[1:]
      pricesUp = delta.copy()
      pricesDown = delta.copy()
      pricesUp[pricesUp < 0] = 0
      pricesDown[pricesDown > 0] = 0
      rollUp = pricesUp.rolling(n).mean()
      rollDown = pricesDown.abs().rolling(n).mean()
      rs = rollUp / rollDown
      rsi = 100.0 - (100.0 / (1.0 + rs))
      return rsi

  # SMA
  df['EMA_5'] = df['Open'].ewm(5).mean().shift()
  df['EMA_10'] = df['Open'].ewm(10).mean().shift()
  df['EMA_20'] = df['Open'].ewm(20).mean().shift()

  # EMA
  df['SMA_5'] = df['Open'].rolling(5).mean().shift()
  df['SMA_10'] = df['Open'].rolling(10).mean().shift()
  df['SMA_20'] = df['Open'].rolling(20).mean().shift()

  # RSI
  df['RSI'] = relative_strength_idx(df).fillna(0)

  # MACD
  EMA_12 = pd.Series(df['Open'].ewm(span=12, min_periods=12).mean())
  EMA_26 = pd.Series(df['Open'].ewm(span=26, min_periods=26).mean())
  df['MACD'] = pd.Series(EMA_12 - EMA_26)
  df['MACD_signal'] = pd.Series(df.MACD.ewm(span=9, min_periods=9).mean())


  df['y'] = df['Open'].shift(1)
  df = df.dropna(axis=0).reset_index(drop=True)


  df_train = df.head(len(df)-len(df2))
  df_valid = df.tail(len(df2))


  features = ['SMA_5','SMA_10','SMA_20','EMA_5','EMA_10','EMA_20','RSI','MACD','MACD_signal']
  model_fbp = Prophet()
  for feature in features:
      model_fbp.add_regressor(feature)

  model_fbp.fit(df_train[['Date', 'y'] + features].rename(columns={'Date': 'ds', 'Open': 'y'}))
  forecast = model_fbp.predict(df_valid[['Date', 'Open'] + features].rename(columns={'Date': 'ds'}))
  df_valid["Forecast_Prophet"] = forecast.yhat.values


  df_valid['new_y'] = df_valid['Open']
  df_valid['Forecast_Prophet_move'] = df_valid['Forecast_Prophet'].shift(-1)
  #df_valid[['new_y', 'Forecast_Prophet_move']].plot()


  predict_price = df_valid['Forecast_Prophet_move'].tolist()
  predict_price.insert(0,df_train['Open'][len(df_train)-1])

  #一階
  own = 0
  answer = []
  for i in range(1,len(predict_price)-1):
    if predict_price[i] - predict_price[i-1] > predict_price[i+1] - predict_price[i]:
      if own == 0:
        own -= 1
        answer.append(-1)
      elif own == 1:
        own -= 1
        answer.append(-1)
      elif own == -1:
        answer.append(0)
    if predict_price[i] - predict_price[i-1] < predict_price[i+1] - predict_price[i]:
      if own == 0:
        own += 1
        answer.append(1)
      elif own == 1:
        answer.append(0)
      elif own == -1:
        own += 1
        answer.append(1)
    elif predict_price[i] - predict_price[i-1] == predict_price[i+1] - predict_price[i]:
      answer.append(0)
  answer.append(0)

  csv_file = open(output, "w")
  for i in answer:
    csv_file.write(str(i)+'\n')
  csv_file.close()


# #print("RMSE of Auto ARIMAX:", np.sqrt(mean_squared_error(df_valid.y, df_valid.Forecast_ARIMAX)))
# print("RMSE of Prophet:", np.sqrt(mean_squared_error(df_valid.y, df_valid.Forecast_Prophet)))
# #print("\nMAE of Auto ARIMAX:", mean_absolute_error(df_valid.y, df_valid.Forecast_ARIMAX))
# print("MAE of Prophet:", mean_absolute_error(df_valid.y, df_valid.Forecast_Prophet))


  

if __name__ == '__main__':
    # You should not modify this part.
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--training',default='training_data.csv',help='input training data file name')
    parser.add_argument('--testing',default='testing_data.csv',help='input testing data file name')
    parser.add_argument('--output',default='output.csv',help='output file name')   
    args = parser.parse_args(args=[])

    run(args.training, args.testing, args.output)
