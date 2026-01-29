"""
Backtest Engine

Compares a strategy (position series) against Buy & Hold.
Reuses tradbot.risk.metrics for all calculations.
"""

import pandas as pd
from tradbot.risk.metrics import (
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_var,
    calculate_annualized_return,
    calculate_annualized_volatility,
)


class Backtest:
    """
    Backtest a strategy vs Buy & Hold.

    Usage:
        bt = Backtest(df_with_strategy)
        bt.run()
        bt.summary()
        bt.get_df()  # DataFrame with cumulative returns for plotting
    """

    def __init__(self, df: pd.DataFrame, strategy_col: str = "Strategy"):
        """
        Args:
            df: DataFrame with 'Close' and a position column (+1/-1/0).
                Positions should already be shifted to avoid look-ahead bias.
            strategy_col: Name of the position column.
        """
        self.df = df.copy()
        self.strategy_col = strategy_col
        self.strategy_metrics = None
        self.buyhold_metrics = None

    def run(self):
        """Run the backtest. Calculates returns and metrics."""
        asset_returns = self.df["Close"].pct_change()
        strategy_returns = asset_returns * self.df[self.strategy_col]

        # Cumulative returns
        self.df["asset_cum_returns"] = (1 + asset_returns).cumprod() - 1
        self.df["strategy_cum_returns"] = (1 + strategy_returns).cumprod() - 1

        # Equity curves
        asset_equity = (1 + asset_returns).cumprod()
        strategy_equity = (1 + strategy_returns).cumprod()

        # Clean NaN
        self.buyhold_metrics = self._calc_metrics(
            asset_returns.dropna(), asset_equity.dropna()
        )
        self.strategy_metrics = self._calc_metrics(
            strategy_returns.dropna(), strategy_equity.dropna()
        )

        return self

    def _calc_metrics(self, returns: pd.Series, equity: pd.Series) -> dict:
        if len(equity) < 2:
            return {
                "total_return": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "var_95": 0.0,
                "ann_return": 0.0,
                "ann_volatility": 0.0,
            }
        return {
            "total_return": float(equity.iloc[-1] - 1),
            "sharpe": calculate_sharpe_ratio(returns),
            "max_drawdown": calculate_max_drawdown(equity),
            "var_95": calculate_var(returns, 0.95),
            "ann_return": calculate_annualized_return(returns),
            "ann_volatility": calculate_annualized_volatility(returns),
        }

    def summary(self) -> dict:
        """
        Return comparison dict after run().

        Returns:
            {"strategy": {metrics}, "buyhold": {metrics}}
        """
        return {
            "strategy": self.strategy_metrics,
            "buyhold": self.buyhold_metrics,
        }

    def get_df(self) -> pd.DataFrame:
        """Return DataFrame with cumulative return columns for plotting."""
        return self.df

    def print_summary(self):
        """Print formatted comparison table."""
        s = self.strategy_metrics
        b = self.buyhold_metrics

        print(f"{'Metric':<20} {'Strategy':>12} {'Buy&Hold':>12}")
        print("-" * 46)
        print(f"{'Total Return':<20} {s['total_return']:>11.2%} {b['total_return']:>11.2%}")
        print(f"{'Sharpe Ratio':<20} {s['sharpe']:>12.3f} {b['sharpe']:>12.3f}")
        print(f"{'Max Drawdown':<20} {s['max_drawdown']:>11.2%} {b['max_drawdown']:>11.2%}")
        print(f"{'VaR 95%':<20} {s['var_95']:>11.4f} {b['var_95']:>11.4f}")
        print(f"{'Ann. Return':<20} {s['ann_return']:>11.2%} {b['ann_return']:>11.2%}")
        print(f"{'Ann. Volatility':<20} {s['ann_volatility']:>11.2%} {b['ann_volatility']:>11.2%}")
