import os
import unittest
from argparse import Namespace
from unittest.mock import patch

from TradingBot import (
    AlpacaSettings,
    StrategyConfig,
    calculate_quantity,
    majority_sentiment,
    normalize_sentiment,
    parse_date,
    parse_date_range,
)


class SentimentHelpersTest(unittest.TestCase):
    def test_normalize_sentiment_extracts_known_labels(self):
        self.assertEqual(normalize_sentiment("The answer is POSITIVE."), "positive")
        self.assertEqual(normalize_sentiment("negative\n"), "negative")
        self.assertEqual(normalize_sentiment("neutral sentiment"), "neutral")

    def test_normalize_sentiment_requires_whole_label_match(self):
        self.assertEqual(normalize_sentiment("notpositively phrased"), "none")

    def test_normalize_sentiment_returns_none_for_unknown_output(self):
        self.assertEqual(normalize_sentiment("mixed"), "none")

    def test_majority_sentiment_uses_most_common_label(self):
        result = majority_sentiment(["positive", "positive", "negative", "none"])
        self.assertEqual(result, "positive")

    def test_majority_sentiment_defaults_for_ties_and_empty_inputs(self):
        self.assertEqual(majority_sentiment(["positive", "negative"]), "neutral")
        self.assertEqual(majority_sentiment([]), "neutral")


class PositionSizingTest(unittest.TestCase):
    def test_calculate_quantity_uses_whole_shares(self):
        self.assertEqual(calculate_quantity(cash=10_000, cash_at_risk=0.5, last_price=432), 11)

    def test_calculate_quantity_never_exceeds_cash_at_risk(self):
        cash = 1_000
        cash_at_risk = 0.5
        last_price = 140

        quantity = calculate_quantity(cash, cash_at_risk, last_price)

        self.assertLessEqual(quantity * last_price, cash * cash_at_risk)

    def test_calculate_quantity_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            calculate_quantity(cash=-1, cash_at_risk=0.5, last_price=100)
        with self.assertRaises(ValueError):
            calculate_quantity(cash=1_000, cash_at_risk=0, last_price=100)
        with self.assertRaises(ValueError):
            calculate_quantity(cash=1_000, cash_at_risk=0.5, last_price=0)


class ConfigurationTest(unittest.TestCase):
    def test_alpaca_settings_loads_from_environment(self):
        env = {
            "ALPACA_API_KEY": "key",
            "ALPACA_API_SECRET": "secret",
            "ALPACA_BASE_URL": "https://example.test/v2",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = AlpacaSettings.from_environment()

        self.assertEqual(settings.api_key, "key")
        self.assertEqual(settings.api_secret, "secret")
        self.assertEqual(settings.base_url, "https://example.test/v2")
        self.assertTrue(settings.paper)

    def test_alpaca_settings_reports_missing_credentials(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "ALPACA_API_KEY, ALPACA_API_SECRET"):
                AlpacaSettings.from_environment()

    def test_parse_date_accepts_iso_dates(self):
        self.assertEqual(parse_date("2024-04-01").strftime("%Y-%m-%d"), "2024-04-01")

    def test_parse_date_range_requires_start_before_end(self):
        start, end = parse_date_range("2024-01-01", "2024-02-01")
        self.assertLess(start, end)

        with self.assertRaisesRegex(ValueError, "start_date must be before end_date"):
            parse_date_range("2024-02-01", "2024-01-01")


class StrategyConfigTest(unittest.TestCase):
    def test_strategy_config_normalizes_symbol_and_exports_parameters(self):
        args = Namespace(symbol=" spy ", cash_at_risk=0.25, lookback_days=5)

        config = StrategyConfig.from_args(args)

        self.assertEqual(config.symbol, "SPY")
        self.assertEqual(
            config.as_parameters(),
            {"symbol": "SPY", "cash_at_risk": 0.25, "lookback_days": 5},
        )

    def test_strategy_config_rejects_invalid_values(self):
        invalid_configs = [
            StrategyConfig(symbol="", cash_at_risk=0.5, lookback_days=3),
            StrategyConfig(symbol="SPY", cash_at_risk=0, lookback_days=3),
            StrategyConfig(symbol="SPY", cash_at_risk=1.5, lookback_days=3),
            StrategyConfig(symbol="SPY", cash_at_risk=0.5, lookback_days=0),
        ]

        for config in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    config.validated()


if __name__ == "__main__":
    unittest.main()
