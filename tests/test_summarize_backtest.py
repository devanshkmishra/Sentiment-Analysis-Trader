import tempfile
import unittest
from pathlib import Path

from scripts.summarize_backtest import format_summary, max_drawdown, summarize


class BacktestSummaryTest(unittest.TestCase):
    def test_max_drawdown_tracks_largest_peak_to_trough_move(self):
        self.assertAlmostEqual(max_drawdown([100, 120, 90, 150]), -0.25)

    def test_summarize_reads_stats_and_filled_trades(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stats_path = root / "stats.csv"
            trades_path = root / "trades.csv"
            stats_path.write_text(
                "\n".join(
                    [
                        "datetime,portfolio_value,cash,return",
                        "2024-01-01,100000,100000,",
                        "2024-01-02,110000,90000,0.10",
                        "2024-01-03,99000,99000,-0.10",
                    ]
                )
            )
            trades_path.write_text(
                "\n".join(
                    [
                        "time,strategy,symbol,side,type,status,filled_quantity",
                        "2024-01-01,MLTrader,SPY,buy,market,new,",
                        "2024-01-01,MLTrader,SPY,buy,market,fill,10",
                        "2024-01-02,MLTrader,SPY,sell,market,fill,5",
                    ]
                )
            )

            summary = summarize(stats_path, trades_path)

        self.assertEqual(summary.start_date, "2024-01-01")
        self.assertEqual(summary.end_date, "2024-01-03")
        self.assertEqual(summary.starting_value, 100000)
        self.assertEqual(summary.ending_value, 99000)
        self.assertAlmostEqual(summary.total_return, -0.01)
        self.assertAlmostEqual(summary.max_drawdown, -0.10)
        self.assertEqual(summary.filled_trades, 2)
        self.assertEqual(summary.buy_fills, 1)
        self.assertEqual(summary.sell_fills, 1)

    def test_format_summary_renders_markdown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stats_path = root / "stats.csv"
            trades_path = root / "trades.csv"
            stats_path.write_text(
                "datetime,portfolio_value,cash,return\n2024-01-01,100,100,\n"
            )
            trades_path.write_text(
                "time,strategy,symbol,side,type,status,filled_quantity\n"
            )
            rendered = format_summary(summarize(stats_path, trades_path))

        self.assertIn("# Backtest Summary", rendered)
        self.assertIn("Total return: 0.00%", rendered)

    def test_summarize_rejects_zero_starting_value(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stats_path = root / "stats.csv"
            trades_path = root / "trades.csv"
            stats_path.write_text(
                "datetime,portfolio_value\n2024-01-01,0\n2024-01-02,100\n"
            )
            trades_path.write_text("status,filled_quantity,side\n")

            with self.assertRaisesRegex(ValueError, "must be greater than zero"):
                summarize(stats_path, trades_path)

    def test_summarize_reports_invalid_portfolio_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stats_path = root / "stats.csv"
            trades_path = root / "trades.csv"
            stats_path.write_text("datetime,portfolio_value\n2024-01-01,unknown\n")
            trades_path.write_text("status,filled_quantity,side\n")

            with self.assertRaisesRegex(ValueError, "Invalid portfolio value on row 2"):
                summarize(stats_path, trades_path)


if __name__ == "__main__":
    unittest.main()
