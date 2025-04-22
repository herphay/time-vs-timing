import yfinance
from datetime import datetime

def main():
    pass

def daily_scapper(ticker):
    today = datetime.today().strftime('%Y-%m-%d')
    existing_data = get_ticker_data(ticker)

    latest_date = existing_data.index.max # TO IMPLEMENT
    # latest_date += 1 day # TO IMPLEMENT

    new_data = get_ticker_history(ticker, start=latest_date)

    new_data.to_db # TO IMPLEMENT

def get_ticker_data(ticker):
    pass

def get_ticker_history(ticker, period=None, start=None, end=None):
    """
    ticker (str): Yahoo Finance stock ticker e.g. ^GSPC for S&P500
    period (str): None = max, Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    start/end (str): YYYY-MM-DD, supercedes period
    """
    return yfinance.Ticker(ticker).history(period=period, start=start, end=end, rounding=True, auto_adjust=False)

if __name__ == '__main__':
    main()