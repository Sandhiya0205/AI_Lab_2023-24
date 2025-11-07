# 🎯 **Experiment No: 13 — Learning: Forecasting Using Supervised Learning (Facebook Prophet)**

### **Date:** 25-10-2025

### **Register Number:** 212223060239

---

## **AIM:**

To develop a Python program that forecasts stock prices using the **Facebook Prophet** model, a supervised learning algorithm for time-series forecasting, and evaluate its accuracy using performance metrics.

---

## **APPARATUS / SOFTWARE REQUIRED:**

* Python 3.x
* Google Colab / Jupyter Notebook
* Libraries: `pandas`, `numpy`, `matplotlib`, `prophet`, `sklearn`

---

## **ALGORITHM:**

### **1. Import Required Libraries**

Import essential Python libraries for:

* Data handling → `pandas`, `numpy`
* Visualization → `matplotlib`
* Modeling → `prophet`
* Evaluation → `sklearn.metrics`

---

### **2. Load Dataset**

* Import the **Amazon stock price dataset (AMZN.csv)**.
* The dataset includes historical data such as `Date`, `Open`, `High`, `Low`, `Close`, and `Adj Close`.
* Display the first few rows using `head()` to verify the dataset.

---

### **3. Data Preparation**

* Prophet requires two specific columns:

  * `ds` → Date (time variable)
  * `y` → Target variable (value to forecast, here “Adj Close”)
* Subset the dataset to include only these columns:

  ```python
  df[['ds', 'y']] = df[['Date', 'Adj Close']]
  df = df[['ds', 'y']]
  ```

---

### **4. Train-Test Split**

* Split the dataset into:

  * **Training data:** until a chosen date (e.g., `"2019-07-21"`)
  * **Testing data:** after that date

  ```python
  split_date = "2019-07-21"
  df_train = df.loc[df.ds <= split_date].copy()
  df_test = df.loc[df.ds > split_date].copy()
  ```

---

### **5. Model Building — Facebook Prophet**

* Initialize and fit the **Prophet model**:

  ```python
  model = fbp.Prophet(daily_seasonality=True)
  model.fit(df_train)
  ```
* Prophet automatically detects seasonality, trend, and holiday effects.

---

### **6. Forecasting**

* Create a **future dataframe** for prediction:

  ```python
  future = model.make_future_dataframe(periods=len(df_test))
  forecast = model.predict(future)
  ```
* The `forecast` output includes:

  * `yhat` → predicted values
  * `yhat_lower` and `yhat_upper` → confidence intervals

---

### **7. Visualization**

* Plot the forecasted vs. actual values using Prophet’s built-in plotting function:

  ```python
  model.plot(forecast)
  model.plot_components(forecast)
  ```
* Observe the **trend**, **weekly**, and **yearly seasonality** components.

---

### **8. Evaluation Metrics**

Compare predicted values with actual stock prices using:

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

y_true = df_test['y'].values
y_pred = forecast.iloc[-len(df_test):]['yhat'].values

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
```

* **MAE (Mean Absolute Error)** — measures the average magnitude of errors.
* **RMSE (Root Mean Square Error)** — penalizes larger errors.
* Smaller values of MAE and RMSE indicate better forecasting performance.

---

### **9. Result Visualization**

* Plot the **training**, **testing**, and **forecasted** stock prices to visually analyze model accuracy.
* Prophet’s interactive plots show how the model fits historical trends and projects future movements.

---

### **10. Result Interpretation**

* Analyze:

  * How close the predicted values are to actual test data.
  * The performance of the model using error metrics.
  * The seasonal and trend insights revealed by Prophet’s decomposition plots.

---

## **PROGRAM:**

```python
# Step 1: Install and import required libraries
!pip install prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import prophet as fbp
from sklearn.metrics import mean_squared_error, mean_absolute_error

plt.style.use('fivethirtyeight')

# Step 2: Load dataset
df = pd.read_csv('AMZN.csv')
df[['ds', 'y']] = df[['Date', 'Adj Close']]
df = df[['ds', 'y']]
df.head()

# Step 3: Train-test split
split_date = "2019-07-21"
df_train = df.loc[df.ds <= split_date].copy()
df_test = df.loc[df.ds > split_date].copy()

# Step 4: Train model
model = fbp.Prophet(daily_seasonality=True)
model.fit(df_train)

# Step 5: Forecast
future = model.make_future_dataframe(periods=len(df_test))
forecast = model.predict(future)

# Step 6: Plot forecast
model.plot(forecast)
plt.title("Amazon Share Price Forecast using Facebook Prophet")
plt.show()

model.plot_components(forecast)

# Step 7: Evaluate performance
y_true = df_test['y'].values
y_pred = forecast.iloc[-len(df_test):]['yhat'].values

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Square Error: {rmse}")
```

---

## **OUTPUT:**

✅ **Sample Dataset:**

| ds         | y       |
| ---------- | ------- |
| 2018-01-02 | 1189.01 |
| 2018-01-03 | 1204.20 |
| 2018-01-04 | 1209.59 |

✅ **Forecast Visualization:**
A line plot showing:

* Blue line → Forecasted stock price
* Black dots → Actual data
* Shaded region → Confidence interval

✅ **Components Plot:**
Displays **trend**, **weekly**, and **yearly** effects on Amazon’s stock price.

✅ **Error Metrics Example:**

```
Mean Absolute Error: 45.83
Root Mean Square Error: 63.27
```

---

## **DETAILED EXPLANATION:**

### **1. Prophet Model**

Prophet is a forecasting tool developed by Facebook designed for business time series with daily observations that display seasonality.

### **2. Working Principle**

It decomposes the time series into:

* **Trend (g(t))** — long-term growth pattern
* **Seasonality (s(t))** — repeating cycles (daily, weekly, yearly)
* **Holidays (h(t))** — special events or anomalies

Model Equation:
[
y(t) = g(t) + s(t) + h(t) + \epsilon_t
]
where (\epsilon_t) is the random error term.

### **3. Training and Forecasting**

The model learns from past data patterns and predicts future values based on identified components. Prophet is robust to missing data and outliers, making it ideal for financial forecasting.

### **4. Evaluation**

MAE and RMSE are used to measure forecast accuracy. Low values indicate better model performance.

---

## **RESULT:**

Thus, the **share price forecasting model** using **Facebook Prophet** was successfully implemented.
The model effectively captured the **trend** and **seasonal components** of the Amazon stock price, generated accurate future predictions, and demonstrated how **supervised learning and time-series modeling** can be applied in real-world financial forecasting.

---

