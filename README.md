# Beta_Lstm_Trader
This project develops a stock price prediction model using Long Short-Term Memory (LSTM)
networks enhanced with financial indicators. We have constructed the LSTM model, integrated a
Convolutional Neural Network (CNN) layer for feature extraction, established a robust data
pipeline, and implemented initial predictions. Our model aims to outperform traditional methods
like ARIMA in accuracy and adaptability, particularly in volatile market conditions.
Additionally, we simulated trading strategies using the model’s predictions, resulting in
consistent profitability across different market scenarios, highlighting the model's effectiveness
in real-world applications.

**1. Introduction**

**Problem Statement:**
The stock market's complexity and unpredictability, driven by numerous factors, challenge
traditional predictive models like ARIMA, which often struggle with the nonlinear nature of
stock prices. In contrast, Long Short-Term Memory (LSTM) models are better equipped to
capture long-term dependencies and nonlinear patterns, making them more effective for stock
price prediction. Implementing an LSTM model for this purpose has the potential to enhance
financial analytics, leading to more informed trading decisions and increased profitability.

****1-2. Related Works**:**

Traditional methods such as Autoregressive Integrated Moving Average (ARIMA) have long
been used in financial forecasting, particularly for stock price prediction. ARIMA is a linear
model that relies on the assumption that past values and their lagged relationships can predict
future values. While effective in some scenarios, ARIMA models often struggle with the
non-linear and volatile nature of stock price movements. As noted in "Time Series Analysis:
Forecasting and Control" by Box et al. (2015), ARIMA’s limitations become apparent in
complex financial environments where the relationships between variables are not linear.
In contrast, Long Short-Term Memory (LSTM) networks, a type of Recurrent Neural Network
(RNN), have emerged as powerful tools for time series prediction, particularly in domains where
long-term dependencies and non-linear patterns are critical. LSTM networks are designed to
remember information for long periods, effectively capturing the temporal dynamics of stock
prices. This makes LSTM particularly suited for financial forecasting, where market conditions
and prices are influenced by long-term trends as well as short-term fluctuations.
Our project builds on the foundation laid by works such as "Long Short-Term Memory" by
Hochreiter & Schmidhuber (1997), which introduced the LSTM architecture, and more recent
applications such as "Deep Learning for Time Series Forecasting" by Jason Brownlee (2019).
These works demonstrate LSTM’s superiority in handling complex, non-linear time series data
compared to traditional models like ARIMA.

Further enhancing the predictive capabilities of LSTM, our approach integrates a Convolutional
Neural Network (CNN) layer for feature extraction before passing the data through the LSTM
layers. The combination of CNN and LSTM has been explored in the context of various
sequential prediction tasks, including financial forecasting. For example, Shi et al. (2015) in their
work "Convolutional LSTM Network: A Machine Learning Approach for Precipitation
Nowcasting," showed how CNNs could effectively capture spatial patterns in data, which, when
combined with LSTM’s temporal modeling, leads to improved predictive accuracy.
In this project, we use the ARIMA model as a baseline to demonstrate the limitations of
traditional methods in handling stock price data. Our LSTM and CNN-LSTM models are
designed to address these limitations by capturing the non-linear and long-term dependencies
inherent in stock market data. By comparing the performance of these models with ARIMA, we
aim to showcase the potential of advanced deep learning techniques in financial forecasting.
The enhancements in our approach, including the integration of financial indicators and the use
of a CNN layer, aim to further refine prediction accuracy and adaptability, particularly under
realistic volatile market conditions. This project not only seeks to validate the effectiveness of
LSTM models in financial analytics but also explores how these models can be integrated into
trading strategies to improve decision-making and profitability.

**2. Data**

**2-1. Dataset Source**

The dataset used in this project consists of historical stock price data from yahoo finance (the SMP 500)

**2-2. Data Preprocessing**

Several preprocessing steps were applied to the dataset to prepare it for model training:

1. Handling Missing Values: Any missing values in the dataset were identified and handled
appropriately. Common strategies include forward-filling or removing the missing data
points, depending on the context.

2. Normalization: The stock prices were normalized using the MinMaxScaler to scale the
features to a range between 0 and 1. This step is essential for ensuring that the models,
particularly neural networks like LSTM, can learn effectively without being biased by the
scale of the input data.

3. Feature Engineering:
○ Lag Features: Lag features were created to provide the model with past data
points as inputs for predicting future values. In this project, lag features were
treated as hyperparameters. Specifically, a window size of **X** days was used,
meaning the model considered the past **X** days of stock prices to predict the next
day’s price. Additionally, **Y** lag features were included, which represent the
closing prices from the previous **Y** days. These lag features help the model capture
recent trends and patterns in the data while avoiding overfitting.

○ Technical Indicators: Several technical indicators were added to the dataset to
enhance the model's ability to predict stock prices. The specific indicators used
were: **Exponential Moving Average (EMA)** with a period of 10 days, **Bollinger Bands** with a period of 20 days **Stochastic Oscillator** with a period of 14 days, Indicators such as **Simple Moving Average (SMA)**, **Relative Strength**
**Index (RSI)**, and **Moving Average Convergence Divergence (MACD)**
were considered but not included in the final mode

**3-4. Model Development**

The core of our project was the development of a Long Short-Term Memory (LSTM) model,
which was enhanced with a Convolutional Neural Network (CNN) layer for feature extraction:

● CNN-LSTM Model: We built a model that first used a CNN layer to extract important
features from the input data, followed by LSTM layers to capture temporal dependencies.
This combination allowed the model to process the time series data more effectively.

● Baseline Comparison with ARIMA: The ARIMA model was implemented as a baseline
to compare the performance of traditional statistical methods with the LSTM-based
approach. The comparison highlighted the limitations of ARIMA in capturing the
non-linear and complex patterns present in the stock prices, which were better handled by
the LSTM model.

**3-5. Hyperparameter Tuning**

To optimize the performance of the LSTM model, we conducted hyperparameter tuning using
Bayesian Optimization. This process helped determine the optimal number of LSTM layers, the
number of neurons per layer, and the learning rate. The tuning process ensured that the model
was both efficient and accurate.

**3-6. Ensemble Learning**

An additional enhancement was the implementation of an ensemble learning approach,
combining the predictions from the LSTM model with those from an ARIMA model:

● ARIMA-LSTM Ensemble: This ensemble method was used to explore whether
combining the strengths of both models could improve prediction accuracy (out of
curiosity). The results provided insights into the complementary nature of statistical and
deep learning models.

**3-7. Trading Simulation**

To assess the practical application of the model's predictions, we developed a trading simulation:
● Simulation Setup: The simulation used the model’s predictions to simulate trading
strategies, including decisions on buying, selling, and reinvesting profits. Key parameters
such as stop-loss percentages and leverage factors were integrated into the simulation to
mimic real-world trading scenarios.

● Performance Metrics: The simulation's outcomes were evaluated based on the final
investment value, comparing the profitability of strategies driven by the LSTM model,
the ARIMA model, and their ensemble

**4-5. Potential Improvements**

● Hyperparameter Tuning: While Bayesian optimization was effective, further
exploration of other tuning methods, such as grid search or random search, could
potentially yield even better results.

● Additional Technical Indicators: Including more diverse technical indicators, such as
MACD or RSI, might enhance the model’s ability to capture different market conditions.

● Alternative Models: Exploring other deep learning architectures, such as GRU (Gated
Recurrent Units) or Transformer models, could provide insights into different ways to
model time series data.

● Refinement of Ensemble Learning: Further research into how best to combine different
models could make ensemble learning a more consistently effective strategy.


5. References
   
● Felbo, Bjarke, et al. “Using Millions of Emoji Occurrences to Learn Any-Domain
Representations for Detecting Sentiment, Emotion and Sarcasm.” arXiv.Org, 7 Oct. 2017,
arxiv.org/abs/1708.00524.
● Medium,
towardsdatascience.com/time-series-prediction-using-lstm-with-tensorflow-2c9b2dfb138
2. Accessed 6 July 2024.
● “Stock Market Prediction Using LSTM Recurrent Neural Network.” Procedia Computer
Science, Elsevier, 14 Apr. 2020,
www.sciencedirect.com/science/article/pii/S1877050920304865.
● “Welcome to the UC Irvine Machine Learning Repository.” UCI Machine Learning
Repository, archive.ics.uci.edu/. Accessed 6 July 2024.
● “Find Open Datasets and Machine Learning Projects.” Kaggle,
www.kaggle.com/datasets. Accessed 6 July 2024.
● “Datasets for Data Science, Machine Learning, AI & Analytics.” KDNuggets,
www.kdnuggets.com/datasets/index.html. Accessed 6 July 2024.
● “Welcome to the UCI Knowledge Discovery in Databases Archive.” UCI KDD Archive,
kdd.ics.uci.edu/. Accessed 6 July 2024.
● George E. P. Box, Gwilym M. Jenkins, Gregory C. Reinsel, and Greta M. Ljung. "Time
Series Analysis: Forecasting and Control." John Wiley & Sons, 2015
