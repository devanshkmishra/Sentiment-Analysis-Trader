"""News sentiment trading strategy for Lumibot and Alpaca.

The module is import-safe: it does not download models, read credentials, or run
backtests until one of the CLI commands is executed.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Sequence


DEFAULT_MODEL_PATH = "Syrinx/llama-2-finance-sentiment"
DEFAULT_SYMBOL = "SPY"
DEFAULT_CASH_AT_RISK = 0.5
DEFAULT_LOOKBACK_DAYS = 3
DEFAULT_START_DATE = "2023-04-01"
DEFAULT_END_DATE = "2024-04-01"
DEFAULT_BASE_URL = "https://paper-api.alpaca.markets/v2"
SENTIMENT_LABELS = ("positive", "neutral", "negative")


@dataclass(frozen=True)
class AlpacaSettings:
    """Credentials and endpoint used by Lumibot and Alpaca."""

    api_key: str
    api_secret: str
    base_url: str = DEFAULT_BASE_URL
    paper: bool = True

    @classmethod
    def from_environment(cls) -> "AlpacaSettings":
        api_key = os.getenv("ALPACA_API_KEY")
        api_secret = os.getenv("ALPACA_API_SECRET")
        base_url = os.getenv("ALPACA_BASE_URL", DEFAULT_BASE_URL)

        missing = [
            name
            for name, value in {
                "ALPACA_API_KEY": api_key,
                "ALPACA_API_SECRET": api_secret,
            }.items()
            if not value
        ]
        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(f"Missing required environment variable(s): {joined}")

        return cls(api_key=api_key or "", api_secret=api_secret or "", base_url=base_url)

    def lumibot_credentials(self) -> dict[str, str | bool]:
        return {
            "API_KEY": self.api_key,
            "API_SECRET": self.api_secret,
            "PAPER": self.paper,
        }


def normalize_sentiment(raw_text: str) -> str:
    """Extract a sentiment label from model output."""

    lowered = raw_text.strip().lower()
    for label in SENTIMENT_LABELS:
        if label in lowered:
            return label
    return "none"


def majority_sentiment(sentiments: Iterable[str], default: str = "neutral") -> str:
    """Return the most common valid sentiment, breaking ties conservatively."""

    counts = Counter(label for label in sentiments if label in SENTIMENT_LABELS)
    if not counts:
        return default

    top_count = max(counts.values())
    winners = {label for label, count in counts.items() if count == top_count}
    if len(winners) > 1:
        return default
    return next(iter(winners))


def calculate_quantity(cash: float, cash_at_risk: float, last_price: float) -> int:
    """Calculate a whole-share position size for the configured risk."""

    if cash < 0:
        raise ValueError("cash must be non-negative")
    if not 0 < cash_at_risk <= 1:
        raise ValueError("cash_at_risk must be greater than 0 and at most 1")
    if last_price <= 0:
        raise ValueError("last_price must be greater than 0")

    return int(round(cash * cash_at_risk / last_price, 0))


class SentimentPredictor:
    """Lazy wrapper around the Hugging Face text-generation pipeline."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH) -> None:
        self.model_path = model_path
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self._pipeline = pipeline(
                task="text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=1,
                do_sample=False,
            )
        return self._pipeline

    def predict_one(self, headline: str) -> str:
        prompt = (
            "Analyze the sentiment of the news headline enclosed in square "
            "brackets, determine if it is positive, neutral, or negative, and "
            'return the corresponding label: "positive", "neutral", or '
            f'"negative". [{headline}] = '
        )
        generated = self._load_pipeline()(prompt)
        answer = generated[0]["generated_text"].split("=")[-1]
        return normalize_sentiment(answer)

    def predict_many(self, headlines: Sequence[str]) -> str:
        return majority_sentiment(self.predict_one(headline) for headline in headlines)


def create_strategy_class(settings: AlpacaSettings, predictor: SentimentPredictor):
    """Create the Lumibot strategy class after runtime dependencies are present."""

    from alpaca_trade_api import REST
    from lumibot.strategies.strategy import Strategy
    from timedelta import Timedelta

    class MLTrader(Strategy):
        def initialize(
            self,
            symbol: str = DEFAULT_SYMBOL,
            cash_at_risk: float = DEFAULT_CASH_AT_RISK,
            lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        ):
            self.symbol = symbol
            self.sleeptime = "24H"
            self.last_trade = None
            self.cash_at_risk = cash_at_risk
            self.lookback_days = lookback_days
            self.api = REST(
                base_url=settings.base_url,
                key_id=settings.api_key,
                secret_key=settings.api_secret,
            )

        def position_sizing(self):
            cash = self.get_cash()
            last_price = self.get_last_price(self.symbol)
            quantity = calculate_quantity(cash, self.cash_at_risk, last_price)
            return cash, last_price, quantity

        def get_dates(self):
            today = self.get_datetime()
            prior = today - Timedelta(days=self.lookback_days)
            return today.strftime("%Y-%m-%d"), prior.strftime("%Y-%m-%d")

        def get_sentiment(self):
            today, prior = self.get_dates()
            news = self.api.get_news(symbol=self.symbol, start=prior, end=today)
            headlines = [event.__dict__["_raw"]["headline"] for event in news]
            return predictor.predict_many(headlines)

        def on_trading_iteration(self):
            cash, last_price, quantity = self.position_sizing()
            if quantity <= 0 or cash <= last_price:
                return

            sentiment = self.get_sentiment()
            if sentiment == "positive":
                if self.last_trade == "sell":
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "buy",
                    type="bracket",
                    take_profit_price=last_price * 1.20,
                    stop_loss_price=last_price * 0.95,
                )
                self.submit_order(order)
                self.last_trade = "buy"
            elif sentiment == "negative":
                if self.last_trade == "buy":
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "sell",
                    type="bracket",
                    take_profit_price=last_price * 0.80,
                    stop_loss_price=last_price * 1.05,
                )
                self.submit_order(order)
                self.last_trade = "sell"

    return MLTrader


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def run_backtest(args: argparse.Namespace) -> None:
    from lumibot.backtesting import YahooDataBacktesting
    from lumibot.brokers import Alpaca

    settings = AlpacaSettings.from_environment()
    predictor = SentimentPredictor(model_path=args.model_path)
    strategy_class = create_strategy_class(settings, predictor)
    broker = Alpaca(settings.lumibot_credentials())

    strategy = strategy_class(
        name="mlstrat",
        broker=broker,
        parameters={
            "symbol": args.symbol,
            "cash_at_risk": args.cash_at_risk,
            "lookback_days": args.lookback_days,
        },
    )
    strategy.backtest(
        YahooDataBacktesting,
        parse_date(args.start_date),
        parse_date(args.end_date),
        parameters={
            "symbol": args.symbol,
            "cash_at_risk": args.cash_at_risk,
            "lookback_days": args.lookback_days,
        },
    )


def run_live(args: argparse.Namespace) -> None:
    from lumibot.brokers import Alpaca
    from lumibot.traders import Trader

    settings = AlpacaSettings.from_environment()
    predictor = SentimentPredictor(model_path=args.model_path)
    strategy_class = create_strategy_class(settings, predictor)
    broker = Alpaca(settings.lumibot_credentials())
    strategy = strategy_class(
        name="mlstrat",
        broker=broker,
        parameters={
            "symbol": args.symbol,
            "cash_at_risk": args.cash_at_risk,
            "lookback_days": args.lookback_days,
        },
    )

    trader = Trader()
    trader.add_strategy(strategy)
    trader.run_all()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a sentiment-driven Alpaca strategy.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_options(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
        command_parser.add_argument("--cash-at-risk", type=float, default=DEFAULT_CASH_AT_RISK)
        command_parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
        command_parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)

    backtest = subparsers.add_parser("backtest", help="Backtest with Yahoo market data.")
    add_common_options(backtest)
    backtest.add_argument("--start-date", default=DEFAULT_START_DATE)
    backtest.add_argument("--end-date", default=DEFAULT_END_DATE)
    backtest.set_defaults(func=run_backtest)

    live = subparsers.add_parser("live", help="Run the strategy against Alpaca paper trading.")
    add_common_options(live)
    live.set_defaults(func=run_live)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
