"""Summarize Lumibot backtest CSV artifacts.

Usage:
    python scripts/summarize_backtest.py \
        --stats logs/MLTrader_2024-09-10_07-09-27_stats.csv \
        --trades logs/MLTrader_2024-09-10_07-30-58_trades.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BacktestSummary:
    start_date: str
    end_date: str
    starting_value: float
    ending_value: float
    total_return: float
    max_drawdown: float
    filled_trades: int
    buy_fills: int
    sell_fills: int


def read_stats(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stats_file:
        rows = [row for row in csv.DictReader(stats_file) if row.get("portfolio_value")]
    if not rows:
        raise ValueError(f"No portfolio rows found in {path}")
    return rows


def read_filled_trades(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as trades_file:
        return [
            row
            for row in csv.DictReader(trades_file)
            if row.get("status") == "fill" and row.get("filled_quantity")
        ]


def max_drawdown(values: list[float]) -> float:
    if not values:
        raise ValueError("At least one portfolio value is required")

    peak = values[0]
    drawdown = 0.0
    for value in values:
        peak = max(peak, value)
        if peak:
            drawdown = min(drawdown, (value - peak) / peak)
    return drawdown


def parse_portfolio_values(stats: list[dict[str, str]], path: Path) -> list[float]:
    values = []
    for row_number, row in enumerate(stats, start=2):
        raw_value = row["portfolio_value"]
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid portfolio value on row {row_number} of {path}: {raw_value!r}"
            ) from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"Portfolio value must be finite and non-negative on row "
                f"{row_number} of {path}"
            )
        values.append(value)
    return values


def summarize(stats_path: Path, trades_path: Path) -> BacktestSummary:
    stats = read_stats(stats_path)
    trades = read_filled_trades(trades_path)
    values = parse_portfolio_values(stats, stats_path)
    starting_value = values[0]
    ending_value = values[-1]
    if starting_value == 0:
        raise ValueError("Starting portfolio value must be greater than zero")
    total_return = (ending_value - starting_value) / starting_value
    buy_fills = sum(1 for row in trades if row.get("side") == "buy")
    sell_fills = sum(1 for row in trades if row.get("side") == "sell")

    return BacktestSummary(
        start_date=stats[0]["datetime"],
        end_date=stats[-1]["datetime"],
        starting_value=starting_value,
        ending_value=ending_value,
        total_return=total_return,
        max_drawdown=max_drawdown(values),
        filled_trades=len(trades),
        buy_fills=buy_fills,
        sell_fills=sell_fills,
    )


def format_summary(summary: BacktestSummary) -> str:
    return "\n".join(
        [
            "# Backtest Summary",
            "",
            f"- Period: {summary.start_date} to {summary.end_date}",
            f"- Starting value: ${summary.starting_value:,.2f}",
            f"- Ending value: ${summary.ending_value:,.2f}",
            f"- Total return: {summary.total_return:.2%}",
            f"- Max drawdown: {summary.max_drawdown:.2%}",
            f"- Filled trades: {summary.filled_trades}",
            f"- Buy fills: {summary.buy_fills}",
            f"- Sell fills: {summary.sell_fills}",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize saved Lumibot CSV logs.")
    parser.add_argument("--stats", type=Path, required=True, help="Path to *_stats.csv")
    parser.add_argument("--trades", type=Path, required=True, help="Path to *_trades.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(format_summary(summarize(args.stats, args.trades)))


if __name__ == "__main__":
    main()
