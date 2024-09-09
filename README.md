# Financial News Sentiment Trading Bot

In this project I have PEFT fine-tuned **LLaMA 2 7B** model using QLoRA for **financial sentiment analysis** and integrated it with an automated trading bot built using **Lumibot** and **Alpaca API**. The bot pulls financial news from yFinance, analyzes sentiment (positive, neutral, or negative), and uses the results to inform trading decisions. The system is capable of backtesting these decisions using historical market data to evaluate the performance.

Try the Trading Bot here: [Google Colab Notebook Link](https://colab.research.google.com/drive/1VNgh9SzLJWpnlOX4_JtxqimZ1qQFkKl1?usp=sharing)

[Click Here to access the Fine tuning Colab Notebook](https://colab.research.google.com/drive/1sZJM6sFJXD6ImOozROOZkUkngBN_g0NK?usp=sharing)

[Click Here to see the fine tuned model repo on Huggingface](https://huggingface.co/Syrinx/llama-2-finance-sentiment)

This repo also contains Logs of the Trading Strategy

## Overview

- **Model Fine-Tuning**: 
   - A **LLaMA 2 7B** model was fine-tuned using **QLoRA** on a dataset of financial news headlines on Kaggle using a P100 GPU, and the fine-tuning process took approximately 1.5 hours.
   - The fine-tuning process utilized **LoRA** (Low-Rank Adaptation) to reduce memory requirements while maintaining performance.
   - The fine-tuned model is quantized using **4-bit precision** for efficient deployment and inference.
   - The training data was extracted from the **Financial PhraseBank** dataset, categorized into positive, neutral, and negative sentiments.
   - The model was optimized for **causal language modeling (CAUSAL_LM)** to ensure it could predict sentiments based on new financial data inputs.
   - This fine-tuned model was uploaded to Hugging Face under the repository [Syrinx/llama-2-finance-sentiment](https://huggingface.co/Syrinx/llama-2-finance-sentiment).

- **Trading Bot**:
   - Built with **Lumibot** and **Alpaca API**, the bot fetches live financial news from **yFinance** and passes it through the sentiment model.
   - Based on the sentiment output, the bot takes appropriate trade actions (buy, sell, hold) on the stock market, specifically using the symbol **SPY** (S&P 500 ETF).
   - The bot uses **YahooDataBacktesting** for backtesting trading strategies from historical data to evaluate performance.



## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install lumibot alpaca-trade-api timedelta transformers torch bitsandbytes
   ```

2. **API Keys**:
   - Get your **Alpaca API** keys from [Alpaca Markets](https://alpaca.markets/), and set them in the environment variables or directly in the script.

3. **Run Backtesting**:
   - The bot is set up to run a backtest over historical data:
     ```python
     strategy.backtest(YahooDataBacktesting, start_date, end_date, parameters={"symbol": "SPY", "cash_at_risk": .5})
     ```

4. **Live Trading** (optional):
   - Uncomment the following code to enable live trading:
     ```python
     trader = Trader()
     trader.add_strategy(strategy)
     trader.run_all()
     ```

## Fine-Tuning Details

The fine-tuning process was performed using the following configurations:
- **LoRA (Low-Rank Adaptation)** with an alpha value of 16 and a dropout rate of 0.1.
- Optimized with **PagedAdamW** and a learning rate of **2e-4**.
- **Gradient Accumulation** to handle batch sizes and **Quantization (4-bit precision)** for efficient model handling.
- The training was conducted over **3 epochs** with a warmup ratio of **0.03** and cosine learning rate scheduling.

## Results

The fine-tuned model achieves competitive accuracy in predicting sentiment, making it suitable for financial trading applications where understanding market sentiment is critical.

100%|██████████| 900/900 [03:51<00:00,  3.89it/s]
Accuracy: 0.847
Accuracy for label 0: 0.890
Accuracy for label 1: 0.870
Accuracy for label 2: 0.780

Classification Report:
              precision    recall  f1-score   support

           0       0.96      0.89      0.92       300
           1       0.73      0.87      0.79       300
           2       0.88      0.78      0.83       300

    accuracy                           0.85       900
   macro avg       0.86      0.85      0.85       900
weighted avg       0.86      0.85      0.85       900


Confusion Matrix:
[[267  31   2]
 [ 10 261  29]
 [  1  65 234]]

---
