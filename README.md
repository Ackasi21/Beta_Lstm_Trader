# Beta_Lstm_Trader

This project develops a stock price prediction model using **Long Short-Term Memory (LSTM)** networks enhanced with **financial indicators**. The model integrates a **Convolutional Neural Network (CNN)** layer for feature extraction, uses a robust data pipeline, and generates stock price predictions.  
It aims to outperform traditional methods such as **ARIMA** in accuracy and adaptability, particularly in volatile market conditions. Trading strategies simulated with these predictions demonstrate consistent profitability across multiple scenarios, highlighting effectiveness in real-world applications.

---

## 1. Introduction

**Problem Statement**  
The stock market’s complexity and unpredictability, influenced by numerous factors, challenge traditional predictive models like ARIMA. These models often struggle with the nonlinear and volatile nature of price movements.  
In contrast, LSTM models are well-suited for capturing long-term dependencies and nonlinear patterns, providing a more powerful approach to financial forecasting. Implementing LSTMs for this purpose has the potential to improve financial analytics, inform trading decisions, and increase profitability.

---

## 2. Related Works

- **Traditional Forecasting (ARIMA):**  
  Relies on lagged values and linear assumptions, effective in some contexts but limited in nonlinear, volatile conditions (Box et al., 2015).  

- **Deep Learning (LSTM):**  
  Introduced by Hochreiter & Schmidhuber (1997), LSTMs capture long-term temporal dependencies and outperform ARIMA in nonlinear environments. Applications such as Brownlee (2019) illustrate their strength in time series forecasting.  

- **CNN-LSTM Hybrid:**  
  Inspired by Shi et al. (2015), where CNNs capture local features before temporal modeling, improving predictive accuracy. This approach is applied here for financial forecasting.  

- **Project Positioning:**  
  ARIMA is used as a baseline, while CNN-LSTM models are developed to capture nonlinear and long-term dependencies. Financial indicators and CNN integration further refine prediction accuracy under volatile market conditions.  
  The models are also tested in trading simulations, validating their application in decision-making and profitability.

---

## 3. Data

### 3.1 Dataset Source  
- Historical stock price data sourced from **Yahoo Finance (S&P 500)**.

### 3.2 Preprocessing  
- **Handling Missing Values:** Forward-fill or removal.  
- **Normalization:** MinMaxScaler applied (range 0–1).  
- **Feature Engineering:**  
  - *Lag Features*: Window size **X** (past X days of prices) and lagged values **Y** included.  
  - *Technical Indicators*: EMA (10), Bollinger Bands (20), Stochastic Oscillator (14).  
  - *Other indicators considered but excluded*: SMA, RSI, MACD.  

---

## 4. Model Development

- **CNN-LSTM Model:**  
  CNN layer for feature extraction, followed by LSTM layers to model sequential dependencies.  

- **ARIMA Baseline:**  
  Implemented to benchmark traditional statistical methods against LSTM-based models.  

### 4.1 Hyperparameter Tuning  
- Bayesian Optimization used to optimize LSTM depth, neuron count, and learning rate.  

### 4.2 Ensemble Learning  
- **ARIMA + LSTM Ensemble** tested to examine complementary strengths of statistical and deep learning approaches.  

### 4.3 Trading Simulation  
- Predictions used to simulate trading decisions (buy, sell, reinvest).  
- Integrated parameters: stop-loss levels, leverage factors.  
- Outcomes measured by final investment value across ARIMA, LSTM, and ensemble approaches.  

---

## 5. Results

- CNN-LSTM achieved **higher accuracy** than ARIMA in capturing nonlinear patterns.  
- Trading simulations showed **consistent profitability** across scenarios.  
- Ensemble results provided exploratory insights into blending statistical and deep learning models.  

---

## 6. Potential Improvements

- Broader hyperparameter tuning (grid/random search).  
- Inclusion of additional indicators (e.g., RSI, MACD).  
- Exploration of alternative architectures (GRU, Transformer).  
- Deeper refinement of ensemble methods.  

---

## 7. References

- Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. *Time Series Analysis: Forecasting and Control*. Wiley, 2015.  
- Hochreiter, S., & Schmidhuber, J. *Long Short-Term Memory*. Neural Computation, 1997.  
- Shi, X., et al. *Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting*. NeurIPS, 2015.  
- Brownlee, J. *Deep Learning for Time Series Forecasting*, 2019.  
- Additional resources: Kaggle Datasets, Yahoo Finance API, UCI ML Repository, KDNuggets Dataset Index.  

---
