# DMAI-HW2-Automated-stock-trading

## OVERVIEW
In this HW, we will implement a very aged prediction problem from the financial field. Given a series of stock prices, including daily open, high, low, and close prices, decide your daily action and make your best profit for the future trading. Can you beat the simple “buy-and-hold” strategy?

## Evaluation Goal
Maximize the profit you gain.
We will use one month’s data as the test data set. Please aim to maximize revenue in 20 days.

## Action Type
The action should be one of these three types:
1 → means to “Buy” the stock. If you short 1 unit, you will return to 0 as the open price in the next day. If you did not have any unit, you will have 1 unit as the open price in the next day. “If you already have 1 unit, your code will be terminated due to the invalid status.“

0 → means to “NoAction”. If you have 1-unit now, hold it. If your slot is available, the status continues. If you short 1 unit, the status continues.

-1 → means to “Sell” the stock. If you hold 1 unit, your will return to 0 as the open price in the next day. If you did not have any unit, we will short 1 unit as the open price in the next day. “If you already short 1 unit, your code will be terminated due to the invalid status.“

So that in any time, your slot status should be:
1 → means you hold 1 unit.
0 → means you don’t hold any unit.
-1 → means you short for 1 unit.

In the final day, if you hold/short the stock, we will force your slot empty as the close price of the final day in the testing period. Finally, your account will be settled and your profit will be calculated.

## 資較集與前處理
使用IBM 過去某5年的股票價格做為資料集，將資料集(csv檔)用pandas.dataframe讀入並賦予日期(Date)，使用簡單移動平均數（Simple Moving Average, SMA）、指數平滑移動平均線（Exponential Smoothing MovingAverage, EMA）分別計算3日、7日、30日的移動平均數，再接著計算指數平滑異同移動平均線（ Moving Average Convergence / Divergence ），將這些計算出的feature加入dataframe當中作為訓練用特徵，再來將資料集中open 做 time-1 的動作，主要是我們想要用滯後資訊來預測下一日的股價，並刪除掉有 null 的資料。

## 模型

使用facebook所開發之prophet模型作為訓練模型，預測出training資料後20日的股價開盤價格並使用testing資料作為驗證，預測結果如下圖:
![下載](https://user-images.githubusercontent.com/48405514/165037613-18ccfff9-c312-40a7-972f-c974e8ae1a4e.png
y為實際股價，Forecast_Prophet為預測之股價。

## 買賣策略
因題目要求在20日內獲得最高的利益，因此買賣策略在不考慮手續費的情形下，要將每一段股價的波動都賺到，才能獲得最大收益。又因買賣的操作只能在隔一日的開盤價格做買賣，買賣的策略要評估往後至少兩日以上才能判定是否要買賣，有以下幾種情形:
1.明天開盤價>後天開盤價
  若手上持有或無股票-->賣出
  若手上負股票>不操作
2.明天開盤價<後天開盤價
  若手上負或無股票-->買入
  若手上持有股票>不操作
3.明天開盤價=後天開盤價
  不操作
透過此買賣策略及可以在所有預測出會漲跌的波段當中賺取價差，獲得最大收益。並將結果輸出至output.csv當中。
