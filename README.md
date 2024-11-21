Bitcoin Price Prediction with Sentiment Analysis
This project leverages historical market data and sentiment analysis of Bitcoin-related Wikipedia edits to predict Bitcoin's price (BTC). By combining time-series analysis with natural language processing (NLP), it examines how market trends and public sentiment influence cryptocurrency prices.

Overview
Key Components:
Historical Market Data: Bitcoin price data is sourced from Yahoo Finance and used for price prediction.
Sentiment Analysis: Wikipedia's Bitcoin page revision history is analyzed for user comment sentiment to incorporate public sentiment into the model.
LSTM Model: A Long Short-Term Memory (LSTM) neural network forecasts future prices by capturing temporal dependencies.
Data Visualization: The performance of the model is evaluated by comparing actual prices against predictions through visual plots.
Dependencies
Ensure the following Python libraries are installed to run the project:

NumPy: For efficient array operations.
Pandas: For data manipulation and analysis.
Matplotlib: For visualizing data.
Yahoo Finance (yfinance): To fetch historical Bitcoin price data.
TensorFlow/Keras: To design and train the LSTM model.
MWClient: For accessing Wikipedia's API.
Transformers: For sentiment analysis using pre-trained models by Hugging Face.
How It Works
1. Data Collection
Market Data: Historical Bitcoin price data in USD is retrieved using Yahoo Finance's API (yfinance).
Sentiment Data: Wikipedia's API (mwclient) provides the Bitcoin page's revision history. Sentiment analysis is performed on user comments using Hugging Face's transformers library, categorizing sentiment as positive or negative and computing sentiment scores.
2. Data Preprocessing
Scaling Prices: Closing prices are normalized between 0 and 1 using MinMaxScaler for training.
Aggregating Sentiment: Sentiment scores are averaged by date, with negative sentiment proportions calculated.
Sliding Window: A sliding window approach uses 60 days of historical prices to predict the price on the 61st day.
3. Model Training
The LSTM model consists of:

Layer 1: LSTM with 50 units, outputting sequences.
Layer 2: LSTM with 50 units.
Layer 3: Dense layer for price prediction.
Dropout layers after each LSTM layer reduce overfitting. The model is trained using the Adam optimizer and mean squared error (MSE) as the loss function.
4. Testing and Prediction
The model predicts Bitcoin prices over a given test range (e.g., January–April 2024). Predictions are compared against actual prices to evaluate accuracy.

5. Visualization
The project generates a plot comparing actual vs. predicted Bitcoin prices, providing insights into the model's performance.

Results
The model predicts Bitcoin's future price, with predictions visualized alongside actual prices. Incorporating sentiment data into the feature set highlights the potential influence of public sentiment on cryptocurrency markets.

How to Run the Project
Clone the repository:
bash
Copy code
git clone https://github.com/yourusername/bitcoin-price-prediction.git  
Install the dependencies:
bash
Copy code
pip install numpy pandas matplotlib yfinance mwclient transformers tensorflow  
Run the script:
bash
Copy code
python bitcoin_price_prediction.py  
Review the generated graphs comparing actual and predicted Bitcoin prices.
Future Enhancements
Potential areas for improvement include:

Additional Sentiment Sources: Analyze sentiment from platforms like Twitter, Reddit, or financial news.
Expanded Feature Engineering: Incorporate financial indicators (e.g., volume, moving averages) or macroeconomic data.
LSTM Optimization: Experiment with hyperparameters, architectures, and optimizers.
Volatility Prediction: Extend the model to predict price volatility, a critical factor in trading.
Conclusion
This project illustrates how LSTM-based time-series models can predict cryptocurrency prices. Integrating sentiment analysis from Wikipedia revisions offers valuable insights into how public sentiment influences Bitcoin's market dynamics.
