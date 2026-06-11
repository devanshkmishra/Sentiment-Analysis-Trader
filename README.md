# Financial News Sentiment Trading Bot

![Result](https://i.imgur.com/9EZZhDR.png)

This project fine-tunes an instruction model with QLoRA for financial headline
sentiment analysis and connects the model to a **Lumibot** strategy backed by
the **Alpaca API**. The current training pipeline defaults to
**Qwen3 4B Instruct**, while the published historical checkpoint uses
**Llama 2 7B**. The strategy reads recent market news, classifies it as
positive, neutral, or negative, and uses that signal to trade **SPY** during
backtests or Alpaca paper trading.

The goal is to demonstrate an end-to-end PEFT workflow for market sentiment, not to present a production trading system or financial advice.

## Links

- Trading bot Colab: [open notebook](https://colab.research.google.com/drive/1VNgh9SzLJWpnlOX4_JtxqimZ1qQFkKl1?usp=sharing)
- Current fine-tuning notebook: [`Qwen3_sentiment_finetune.ipynb`](Qwen3_sentiment_finetune.ipynb)
- Historical fine-tuned model: [Syrinx/llama-2-finance-sentiment](https://huggingface.co/Syrinx/llama-2-finance-sentiment)
- Saved result summary: [docs/BACKTEST_SUMMARY.md](docs/BACKTEST_SUMMARY.md)

## What Is Included

- `TradingBot.py`: import-safe CLI for running backtests or paper trading.
- `finetune_sentiment.py`: reproducible QLoRA training and evaluation CLI.
- `Qwen3_sentiment_finetune.ipynb`: small GPU-notebook entry point for the CLI.
- `TradingBot.ipynb`: original notebook workflow.
- `logs/`: saved Lumibot backtest CSV and HTML artifacts.
- `scripts/summarize_backtest.py`: utility for summarizing saved backtest logs.
- `tests/`: fast unit tests for sentiment parsing, position sizing, config loading, and log summaries.

## Setup

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   For model fine-tuning on a CUDA machine, install the training extras instead:

   ```bash
   pip install -r requirements-finetune.txt
   ```

3. Configure Alpaca paper-trading credentials:

   ```bash
   cp .env.example .env
   ```

   Then edit `.env` and export the variables before running the bot:

   ```bash
   export ALPACA_API_KEY=your_alpaca_paper_key
   export ALPACA_API_SECRET=your_alpaca_paper_secret
   export ALPACA_BASE_URL=https://paper-api.alpaca.markets/v2
   ```

## Run A Backtest

```bash
python TradingBot.py backtest \
  --symbol SPY \
  --cash-at-risk 0.5 \
  --lookback-days 3 \
  --start-date 2023-04-01 \
  --end-date 2024-04-01
```

The script loads the Hugging Face model lazily, so importing `TradingBot.py` for tests or helper functions will not download model weights or contact Alpaca.

## Run Paper Trading

```bash
python TradingBot.py live --symbol SPY --cash-at-risk 0.5
```

Use Alpaca paper credentials unless you have deliberately reviewed and changed the strategy for live trading risk.

## Summarize Saved Logs

```bash
python scripts/summarize_backtest.py \
  --stats logs/MLTrader_2024-09-10_07-09-27_stats.csv \
  --trades logs/MLTrader_2024-09-10_07-30-58_trades.csv
```

The current saved logs summarize to:

- Total return: 19.64%
- Max drawdown: -47.68%
- Filled trades: 96

## Test

```bash
python3 -m unittest discover -s tests
```

The tests are intentionally lightweight and do not require Lumibot, Alpaca credentials, or the LLaMA model.

## Fine-Tune A Model

Run on a CUDA GPU:

```bash
python finetune_sentiment.py \
  --model-name Qwen/Qwen3-4B-Instruct-2507 \
  --dataset-config sentences_allagree \
  --output-dir artifacts/qwen3-finance-sentiment
```

The default pipeline provides:

- An ungated, Apache-2.0 **Qwen3 4B Instruct** base model.
- The Hugging Face `takala/financial_phrasebank` dataset without manual uploads.
- Deterministic, label-stratified train, validation, and test splits.
- 4-bit NF4 quantization with double quantization and bf16 where supported.
- LoRA across linear layers, gradient checkpointing, and paged AdamW.
- Best-checkpoint selection using validation loss.
- Greedy exact-label test evaluation with accuracy, macro F1, per-label metrics,
  and invalid-output counts saved to `test_metrics.json`.
- A saved `training_config.json` for reproducibility.

Useful overrides:

```bash
# More examples with lower annotator agreement
python finetune_sentiment.py --dataset-config sentences_75agree

# Lower-memory GPU
python finetune_sentiment.py \
  --batch-size 1 \
  --gradient-accumulation-steps 16

# Try another compatible causal instruction model
python finetune_sentiment.py --model-name Qwen/Qwen2.5-3B-Instruct
```

The adapter and tokenizer are written to `--output-dir`. The historical trading
bot continues to default to the published Llama 2 checkpoint; pass
`--model-path` to backtest a merged or published replacement model.

## Historical Model Evaluation

The original Llama 2 fine-tuning run reported the following held-out sentiment
results. These are retained for reference and are not claims about the new
Qwen3 pipeline until a new training run is completed:

```text
Accuracy: 0.847
Accuracy for label 0: 0.890
Accuracy for label 1: 0.870
Accuracy for label 2: 0.780
```

```text
              precision    recall  f1-score   support

           0       0.96      0.89      0.92       300
           1       0.73      0.87      0.79       300
           2       0.88      0.78      0.83       300

    accuracy                           0.85       900
   macro avg       0.86      0.85      0.85       900
weighted avg       0.86      0.85      0.85       900
```

Confusion matrix:

```text
[[267  31   2]
 [ 10 261  29]
 [  1  65 234]]
```

## Results

The saved strategy results compare the sentiment-based strategy against SPY between January 2020 and April 2024. The strategy outperformed early in the test window and later suffered from a sharp drawdown, which suggests the sentiment signal needs stronger confidence thresholds and risk management before serious use.

![Metrics](https://i.imgur.com/Ya8QInW.png)
