# Time Series Forecasting for Portfolio Management Optimization

## Business Objective

Guide Me in Finance (GMF) Investments is a financial advisory firm that utilizes advanced data-driven insights for portfolio management. The objective is to predict market trends, optimize asset allocation, and enhance portfolio performance using time series forecasting models.

As a Financial Analyst at GMF Investments, your role is to:
- Extract and preprocess financial data.
- Develop and evaluate forecasting models.
- Provide insights on future market movements.
- Recommend portfolio adjustments to optimize returns while minimizing risks.

## Data

We use historical financial data for three key assets:
- **Tesla (TSLA):** High-growth, high-risk stock.
- **Vanguard Total Bond Market ETF (BND):** Stable, low-risk bond ETF.
- **S&P 500 ETF (SPY):** Diversified, moderate-risk market exposure.


# Task 1: Preprocess and Explore the Data

### 1. Load, Clean, and Understand the Data
- Fetch historical stock prices using `YFinance`.
- Check for missing values and handle them through interpolation or removal.
- Normalize/scale the data as required for machine learning models.
- Convert date columns to datetime format for time series analysis.

### 2. Exploratory Data Analysis (EDA)
- **Visualizations:**
  - Plot closing prices over time.
  - Compute and plot **daily percentage changes** to observe volatility.
  - Perform **rolling mean and standard deviation** analysis to track short-term trends.
- **Outlier Detection:**
  - Identify days with **unusually high or low** returns.
  - Analyze anomalies using statistical measures (e.g., Z-score, IQR method).
- **Seasonality & Trend Analysis:**
  - Decompose the time series into **trend, seasonal, and residual** components using `statsmodels`.
  - Identify cyclical patterns in stock movements.
- **Volatility Analysis:**
  - Compute rolling standard deviations to assess risk.
  - Use **Value at Risk (VaR)** and **Sharpe Ratio** to evaluate stock risk levels.

---

# Task 2: Develop Time Series Forecasting Models

This task involves building predictive models for Tsla's stock prices. You can choose between:

- **ARIMA (AutoRegressive Integrated Moving Average):** Best for univariate time series with no seasonality.
- **SARIMA (Seasonal ARIMA):** Incorporates seasonality into ARIMA models.
- **LSTM (Long Short-Term Memory):** Deep learning-based model for capturing long-term dependencies in time series data.

### 1. Data Preparation
- Split the dataset into **training and testing sets**.
- Define the target variable (closing price).

### 2. Model Training and Forecasting
- Train **ARIMA/SARIMA/LSTM** models on the training dataset.
- Generate forecasts for the **next 6-12 months**.
- Compare predicted values with actual test data.

### 3. Model Evaluation
- Calculate performance metrics:
  - **Mean Absolute Error (MAE)**
  - **Root Mean Squared Error (RMSE)**
  - **Mean Absolute Percentage Error (MAPE)**
- Visualize forecast results with **confidence intervals**.

### 4. Optimization and Fine-Tuning
- Use **grid search** to optimize ARIMA/SARIMA parameters `(p, d, q)`.
- Adjust **LSTM hyperparameters** (number of layers, neurons, learning rate).
- Perform **backtesting** to assess model robustness.

---

## How to Run the Code
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the data preprocessing and EDA script:
   ```bash
   python preprocess_data.py
   ```
4. Train the forecasting model:
   ```bash
   python train_model.py --model arima
   ```
5. Evaluate the results and visualize:
   ```bash
   python evaluate.py
   ```

## References
- [ARIMA for Time Series Forecasting](https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/)
- [Portfolio Optimization in Python](https://www.machinelearningplus.com/machine-learning/portfolio-optimization-python-example/)
- [Time Series Forecasting Guide](https://www.geeksforgeeks.org/time-series-analysis-and-forecasting/)
