"""
Execution - TWS Paper Trading Interface

Connect to Interactive Brokers TWS for quotes, historical data, and order execution.
"""

import streamlit as st
import pandas as pd
from components.kpi_cards import render_connection_badge
from components.charts import candlestick_chart

st.header("Execution")

# --- TWS Connection ---
try:
    from tradbot.execution import TWSConnector
    tws_available = True
except ImportError:
    tws_available = False
    st.warning("ib_insync not installed. Run `pip install ib_insync` to enable TWS integration.")

# --- Sidebar ---
with st.sidebar:
    st.subheader("TWS Connection")
    host = st.text_input("Host", value="127.0.0.1", key="tws_host")
    port = st.number_input("Port", value=7497, step=1, key="tws_port",
                           help="7497 = Paper Trading, 7496 = Live")
    client_id = st.number_input("Client ID", value=1, step=1, key="tws_client_id")

# Connection state
if "tws_connected" not in st.session_state:
    st.session_state.tws_connected = False
if "tws_connector" not in st.session_state:
    st.session_state.tws_connector = None

# Connect/Disconnect buttons
col_conn1, col_conn2, col_conn3 = st.columns([1, 1, 2])

with col_conn1:
    if st.button("Connect", disabled=not tws_available):
        try:
            tws = TWSConnector(host=host, port=int(port), client_id=int(client_id))
            if tws.connect():
                st.session_state.tws_connector = tws
                st.session_state.tws_connected = True
                st.rerun()
            else:
                st.error("Connection failed. Is TWS running with API enabled?")
        except Exception as e:
            st.error(f"Connection error: {e}")

with col_conn2:
    if st.button("Disconnect", disabled=not st.session_state.tws_connected):
        if st.session_state.tws_connector:
            st.session_state.tws_connector.disconnect()
        st.session_state.tws_connected = False
        st.session_state.tws_connector = None
        st.rerun()

with col_conn3:
    render_connection_badge(st.session_state.tws_connected)

st.divider()

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["Quote Lookup", "Historical Data", "Order Entry"])

tws = st.session_state.tws_connector

# === Tab 1: Quote Lookup ===
with tab1:
    st.subheader("Live Quote")

    if not st.session_state.tws_connected:
        st.info("Connect to TWS to access live quotes")
    else:
        quote_symbol = st.text_input("Symbol", value="AAPL", key="quote_symbol")
        if st.button("Get Quote"):
            try:
                with st.spinner("Fetching quote..."):
                    ticker = tws.get_quote(quote_symbol)

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Bid", f"${ticker.bid:.2f}" if ticker.bid else "N/A")
                col2.metric("Ask", f"${ticker.ask:.2f}" if ticker.ask else "N/A")
                col3.metric("Last", f"${ticker.last:.2f}" if ticker.last else "N/A")
                col4.metric("Volume", f"{ticker.volume:,.0f}" if ticker.volume else "N/A")

                tws.stop_market_data(ticker)
            except Exception as e:
                st.error(f"Quote failed: {e}")

# === Tab 2: Historical Data ===
with tab2:
    st.subheader("Historical Data")

    if not st.session_state.tws_connected:
        st.info("Connect to TWS to access historical data")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            hist_symbol = st.text_input("Symbol", value="AAPL", key="hist_symbol")
        with col2:
            duration = st.selectbox("Duration", ["5 D", "1 M", "3 M", "6 M", "1 Y"],
                                    index=0, key="hist_duration")
        with col3:
            bar_size = st.selectbox("Bar Size",
                                    ["1 min", "5 mins", "15 mins", "1 hour", "1 day"],
                                    index=3, key="hist_bar_size")

        if st.button("Load Historical"):
            try:
                with st.spinner("Loading historical data..."):
                    hist_df = tws.get_historical(hist_symbol, duration, bar_size)

                if hist_df.empty:
                    st.warning("No data returned")
                else:
                    st.success(f"{len(hist_df)} bars loaded")

                    # Rename columns to match chart expectations
                    col_map = {"open": "Open", "high": "High", "low": "Low",
                               "close": "Close", "volume": "Volume"}
                    hist_df = hist_df.rename(columns=col_map)

                    if all(c in hist_df.columns for c in ["Open", "High", "Low", "Close"]):
                        fig = candlestick_chart(hist_df, f"{hist_symbol} - {bar_size}")
                        st.plotly_chart(fig, use_container_width=True)

                    st.dataframe(hist_df.tail(50), use_container_width=True)

            except Exception as e:
                st.error(f"Historical data failed: {e}")

# === Tab 3: Order Entry ===
with tab3:
    st.subheader("Paper Trading")

    if not st.session_state.tws_connected:
        st.info("Connect to TWS to place orders")
    else:
        st.markdown("**Place Order**")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            order_symbol = st.text_input("Symbol", value="AAPL", key="order_symbol")
        with col2:
            order_qty = st.number_input("Quantity", value=10, min_value=1, key="order_qty")
        with col3:
            order_action = st.selectbox("Action", ["BUY", "SELL"], key="order_action")
        with col4:
            order_type = st.selectbox("Type", ["Market", "Limit", "Stop"], key="order_type")

        limit_price = None
        if order_type in ["Limit", "Stop"]:
            limit_price = st.number_input("Price", value=150.0, step=0.01, key="order_price")

        if st.button("Place Order", type="primary"):
            try:
                with st.spinner("Placing order..."):
                    if order_type == "Market":
                        trade = tws.market_order(order_symbol, order_qty, order_action)
                    elif order_type == "Limit":
                        trade = tws.limit_order(order_symbol, order_qty, limit_price, order_action)
                    else:
                        trade = tws.stop_order(order_symbol, order_qty, limit_price, order_action)

                if trade:
                    st.success(f"Order placed: {order_action} {order_qty} {order_symbol} "
                               f"({order_type}) - Status: {trade.orderStatus.status}")
                else:
                    st.error("Order failed")
            except Exception as e:
                st.error(f"Order error: {e}")

        st.divider()

        # Open Orders
        st.markdown("**Open Orders**")
        if st.button("Refresh Orders"):
            try:
                orders = tws.get_open_orders()
                if orders:
                    order_data = [{
                        "Symbol": o.contract.symbol if hasattr(o, "contract") else "N/A",
                        "Action": o.action,
                        "Quantity": o.totalQuantity,
                        "Type": o.orderType,
                        "Status": o.status if hasattr(o, "status") else "N/A",
                    } for o in orders]
                    st.dataframe(pd.DataFrame(order_data), use_container_width=True, hide_index=True)
                else:
                    st.info("No open orders")
            except Exception as e:
                st.error(f"Could not fetch orders: {e}")

        st.divider()

        # Positions
        st.markdown("**Positions**")
        if st.button("Refresh Positions"):
            try:
                positions = tws.get_positions()
                if positions:
                    pos_data = [{
                        "Symbol": p.contract.symbol,
                        "Position": p.position,
                        "Avg Cost": f"${p.avgCost:.2f}",
                        "Market Value": f"${p.position * p.avgCost:.2f}",
                    } for p in positions]
                    st.dataframe(pd.DataFrame(pos_data), use_container_width=True, hide_index=True)
                else:
                    st.info("No open positions")
            except Exception as e:
                st.error(f"Could not fetch positions: {e}")
