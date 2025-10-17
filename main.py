from datetime import date, timedelta
import streamlit as st
from data_fetcher import fetch_and_save_stock_data
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import zscore

st.set_page_config(
    page_title="Stock Market Anomaly Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# App Title & Description
# -----------------------------
st.title("Stock Market Anomaly Detection")

st.header("Key Concepts")
st.write(
    """
Stock market anomaly detection identifies unusual patterns or behaviors in stock data that deviate significantly from the expected norm.
These events are unexpected and can lead to significant price movements or unusual trading volumes.
"""
)
st.write("We will collect real-time stock market data using the yfinance API.")

# -----------------------------
# Data Fetching Section
# -----------------------------
st.header("📊 Stock Market Data Fetcher")

# Top 20 Indian company tickers (NSE)
top_20_india_tickers = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "BAJFINANCE.NS", "BHARTIARTL.NS",
    "ITC.NS", "HCLTECH.NS", "LT.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "WIPRO.NS", "AXISBANK.NS", "TITAN.NS", "NESTLEIND.NS"
]

# Multi-select dropdown
tickers = st.multiselect(
    "Select Stock Tickers:",
    options=top_20_india_tickers,
    default=["RELIANCE.NS", "TCS.NS"]
)

# Date inputs
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", date.today() - timedelta(days=365))
with col2:
    end_date = st.date_input("End Date", date.today())

# Auto-adjust checkbox
auto_adjust = st.checkbox("Auto-adjust OHLC prices?", value=False)

# File name input
file_name = st.text_input("Enter file name for CSV:", "stock_data")

# Initialize DataFrame
stock_data = pd.DataFrame()

# -----------------------------
# Fetch Button
# -----------------------------
if st.button("Fetch and Save Data"):
    if not tickers:
        st.error("Please select at least one ticker.")
    else:
        csv_file_path = f"{file_name}.csv"

        # Fetch stock data
        stock_data = fetch_and_save_stock_data(
            tickers=tickers,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            file_path=csv_file_path,
            auto_adjust=auto_adjust
        )

        st.success(f"✅ Data saved to {csv_file_path}")

        # Convert 'Date' to datetime and set as index
        stock_data['Date'] = pd.to_datetime(stock_data['Date'], errors='coerce')
        stock_data = stock_data.dropna(subset=['Date'])
        stock_data.set_index('Date', inplace=True)
        stock_data.sort_index(inplace=True)

        # Show preview
        st.dataframe(stock_data.head())

        # Download CSV
        st.download_button(
            label="⬇️ Download CSV",
            data=stock_data.to_csv().encode("utf-8"),
            file_name=csv_file_path,
            mime="text/csv",
        )

        # -----------------------------
        # Determine price column
        # -----------------------------
        price_col = 'Adj Close' if 'Adj Close' in stock_data.columns else 'Close'

        # -----------------------------
        # Price Plots (Interactive Plotly)
        # -----------------------------
        st.markdown("### 📊 Comparison of Opening Prices Over Time")
        fig_open = px.line(
            stock_data,
            x=stock_data.index,
            y='Open',
            color='Ticker',
            title='Open Prices Comparison'
        )
        st.plotly_chart(fig_open, use_container_width=True)

        st.markdown(f"### 📈 Comparison of {price_col} Over Time")
        fig_price = px.line(
            stock_data,
            x=stock_data.index,
            y=price_col,
            color='Ticker',
            title=f'{price_col} Prices Comparison'
        )
        st.plotly_chart(fig_price, use_container_width=True)

        st.markdown("### 📊 Trading Volume Over Time")
        fig_volume = px.line(
            stock_data,
            x=stock_data.index,
            y='Volume',
            color='Ticker',
            title='Trading Volume Comparison'
        )
        st.plotly_chart(fig_volume, use_container_width=True)

        # -----------------------------
        # Anomaly Detection
        # -----------------------------
        st.markdown("### 🚨 Detect Anomalies in Prices and Volume")

        def detect_anomalies(df, column, threshold=2):
            df_copy = df.copy()
            df_copy['Z-score'] = zscore(df_copy[column])
            anomalies = df_copy[abs(df_copy['Z-score']) > threshold]
            return anomalies

        anomalies_adj_close = pd.DataFrame()
        anomalies_volume = pd.DataFrame()

        for ticker in stock_data['Ticker'].unique():
            subset = stock_data[stock_data['Ticker'] == ticker]
            adj_anomalies = detect_anomalies(subset, price_col)
            vol_anomalies = detect_anomalies(subset, 'Volume')
            anomalies_adj_close = pd.concat([anomalies_adj_close, adj_anomalies])
            anomalies_volume = pd.concat([anomalies_volume, vol_anomalies])

        st.write(f"Total anomalies in {price_col}: {len(anomalies_adj_close)}")
        st.write(f"Total anomalies in volume: {len(anomalies_volume)}")

        # -----------------------------
        # Anomaly Plots (Interactive Plotly)
        # -----------------------------
        st.markdown("### 📊 Anomaly Visualization by Ticker")
        for ticker in stock_data['Ticker'].unique():
            st.markdown(f"#### {ticker}")
            subset = stock_data[stock_data['Ticker'] == ticker]
            adj_anom = anomalies_adj_close[anomalies_adj_close['Ticker'] == ticker]
            vol_anom = anomalies_volume[anomalies_volume['Ticker'] == ticker]

            fig = go.Figure()

            # Price line
            fig.add_trace(go.Scatter(
                x=subset.index,
                y=subset[price_col],
                mode='lines',
                name=price_col
            ))

            # Price anomalies
            fig.add_trace(go.Scatter(
                x=adj_anom.index,
                y=adj_anom[price_col],
                mode='markers',
                marker=dict(color='red', size=10),
                name='Price Anomalies'
            ))

            # Volume line
            fig.add_trace(go.Scatter(
                x=subset.index,
                y=subset['Volume'],
                mode='lines',
                name='Volume',
                yaxis='y2'
            ))

            # Volume anomalies
            fig.add_trace(go.Scatter(
                x=vol_anom.index,
                y=vol_anom['Volume'],
                mode='markers',
                marker=dict(color='orange', size=10),
                name='Volume Anomalies',
                yaxis='y2'
            ))

            # Layout with secondary y-axis
            fig.update_layout(
                title=f"{ticker} Price & Volume Anomalies",
                xaxis_title='Date',
                yaxis_title=price_col,
                yaxis2=dict(
                    title='Volume',
                    overlaying='y',
                    side='right'
                ),
                legend=dict(orientation="h")
            )

            st.plotly_chart(fig, use_container_width=True)

        # -----------------------------
        # Correlation Matrix (Interactive Heatmap)
        # -----------------------------
        st.markdown("### 🔗 Correlation Matrix of Anomalies Across Tickers")

        # Prepare anomaly indicators
        all_anomalies_adj_close = anomalies_adj_close[['Ticker']].copy()
        all_anomalies_adj_close['Adj Close Anomaly'] = 1

        all_anomalies_volume = anomalies_volume[['Ticker']].copy()
        all_anomalies_volume['Volume Anomaly'] = 1

        adj_close_pivot = all_anomalies_adj_close.pivot_table(
            index=all_anomalies_adj_close.index, columns='Ticker', values='Adj Close Anomaly', fill_value=0
        )

        volume_pivot = all_anomalies_volume.pivot_table(
            index=all_anomalies_volume.index, columns='Ticker', values='Volume Anomaly', fill_value=0
        )

        combined_anomalies = pd.concat(
            [adj_close_pivot.add_prefix('Price_'), volume_pivot.add_prefix('Volume_')],
            axis=1
        )

        correlation_matrix = combined_anomalies.corr()

        fig_corr = px.imshow(
            correlation_matrix,
            text_auto=False,
            color_continuous_scale='RdBu_r',
            title="Correlation of Price and Volume Anomalies Across Tickers"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # -----------------------------
        # Risk Analysis
        # -----------------------------
        st.markdown("### ⚠️ Risk Analysis Based on Anomalies")

        if not stock_data.empty and not anomalies_adj_close.empty and not anomalies_volume.empty:
            adj_close_risk = anomalies_adj_close.groupby('Ticker')['Z-score'].apply(lambda x: abs(x).mean())
            volume_risk = anomalies_volume.groupby('Ticker')['Z-score'].apply(lambda x: abs(x).mean())
            total_risk = adj_close_risk.add(volume_risk, fill_value=0)
            risk_rating = (total_risk - total_risk.min()) / (total_risk.max() - total_risk.min())

            risk_df = pd.DataFrame({
                'Ticker': risk_rating.index,
                'Relative Risk Rating': risk_rating.values
            }).sort_values(by='Relative Risk Rating', ascending=False)

            st.dataframe(risk_df.style.background_gradient(cmap='Reds'))

            fig_risk = px.bar(
                risk_df,
                x='Ticker',
                y='Relative Risk Rating',
                text='Relative Risk Rating',
                color='Relative Risk Rating',
                color_continuous_scale='Reds',
                title="Relative Risk Ratings Across Selected Tickers"
            )
            fig_risk.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig_risk.update_layout(yaxis=dict(range=[0, 1.1]))
            st.plotly_chart(fig_risk, use_container_width=True)
