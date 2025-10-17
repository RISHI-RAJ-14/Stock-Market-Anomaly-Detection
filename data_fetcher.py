import pandas as pd
import yfinance as yf

def fetch_and_save_stock_data(
    tickers, start_date, end_date, file_path="stock_data.csv", auto_adjust=False
):
    """
    Fetch historical stock data for given tickers and save as CSV.
    
    Parameters:
        tickers (list): List of stock ticker symbols, e.g., ['AAPL','MSFT']
        start_date (str): Start date in 'YYYY-MM-DD'
        end_date (str): End date in 'YYYY-MM-DD'
        file_path (str): Path to save CSV
        auto_adjust (bool): Whether to auto-adjust OHLC prices (if True, 'Adj Close' removed)
    
    Returns:
        pd.DataFrame: Flat stock data with Date, Ticker, Open, High, Low, Close, Adj Close, Volume
    """

    # Download data from Yahoo Finance
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        progress=False,
        group_by='ticker',
        auto_adjust=auto_adjust
    )

    all_data = []

    for ticker in tickers:
        if ticker in data.columns.levels[0]:  # Multi-ticker
            df = data[ticker].copy()
        else:  # Single ticker
            df = data.copy()
        df['Ticker'] = ticker
        df['Date'] = df.index
        all_data.append(df.reset_index(drop=True))

    stock_data = pd.concat(all_data, ignore_index=True)

    # Decide which columns exist based on auto_adjust
    cols = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
    if not auto_adjust and 'Adj Close' in stock_data.columns:
        cols.insert(-1, 'Adj Close')  # Add Adj Close before Volume

    stock_data = stock_data[cols]

    # Save CSV
    #stock_data.to_csv(file_path, index=False)
    print(f"✅ Data saved to {file_path}")

    return stock_data
