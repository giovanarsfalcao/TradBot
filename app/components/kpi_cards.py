"""
KPI metric card helpers for the dashboard.
"""

import streamlit as st

COLORS = {
    "green": "#00d4aa",
    "red": "#ff4757",
    "blue": "#1e88e5",
    "orange": "#ffa502",
    "gray": "#747d8c",
}


def render_kpi(label: str, value: str, delta: str = None,
               delta_color: str = "normal"):
    """Render a single KPI metric using Streamlit's metric widget."""
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


def render_kpi_row(metrics: list[dict]):
    """
    Render a row of KPI cards.

    Args:
        metrics: list of dicts with keys: label, value, delta (optional), delta_color (optional)
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            st.metric(
                label=m["label"],
                value=m["value"],
                delta=m.get("delta"),
                delta_color=m.get("delta_color", "normal"),
            )


def render_signal_badge(signal: str, reason: str = ""):
    """
    Render a colored signal badge (BUY/SELL/HOLD).
    """
    color_map = {
        "BUY": COLORS["green"],
        "SELL": COLORS["red"],
        "HOLD": COLORS["gray"],
        "LONG": COLORS["green"],
        "SHORT": COLORS["red"],
        "FLAT": COLORS["gray"],
    }
    color = color_map.get(signal.upper(), COLORS["gray"])
    st.markdown(
        f'<div style="background-color:{color}; color:white; padding:12px 20px; '
        f'border-radius:8px; text-align:center; font-size:1.2em; font-weight:bold;">'
        f'{signal.upper()}</div>',
        unsafe_allow_html=True,
    )
    if reason:
        st.caption(reason)


def render_connection_badge(connected: bool):
    """Render a connection status badge."""
    if connected:
        st.markdown(
            '<div style="background-color:#00d4aa; color:white; padding:8px 16px; '
            'border-radius:8px; text-align:center; font-weight:bold;">'
            'Connected</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="background-color:#ff4757; color:white; padding:8px 16px; '
            'border-radius:8px; text-align:center; font-weight:bold;">'
            'Not Connected</div>',
            unsafe_allow_html=True,
        )


def render_metric_table(strategy_metrics: dict, buyhold_metrics: dict):
    """
    Render a comparison table of strategy vs buy & hold metrics.
    """
    import pandas as pd

    rows = [
        ("Total Return", f"{strategy_metrics['total_return']:.2%}",
         f"{buyhold_metrics['total_return']:.2%}"),
        ("Sharpe Ratio", f"{strategy_metrics['sharpe']:.3f}",
         f"{buyhold_metrics['sharpe']:.3f}"),
        ("Max Drawdown", f"{strategy_metrics['max_drawdown']:.2%}",
         f"{buyhold_metrics['max_drawdown']:.2%}"),
        ("VaR 95%", f"{strategy_metrics['var_95']:.4f}",
         f"{buyhold_metrics['var_95']:.4f}"),
        ("Ann. Return", f"{strategy_metrics['ann_return']:.2%}",
         f"{buyhold_metrics['ann_return']:.2%}"),
        ("Ann. Volatility", f"{strategy_metrics['ann_volatility']:.2%}",
         f"{buyhold_metrics['ann_volatility']:.2%}"),
    ]

    table = pd.DataFrame(rows, columns=["Metric", "Strategy", "Buy & Hold"])
    st.dataframe(table, use_container_width=True, hide_index=True)
