import yfinance
from datetime import datetime
import sqlite3

def main() -> None:
    pass

def ticker_scapper(ticker: str):
    today = datetime.today().strftime('%Y-%m-%d')
    existing_data = gpull_ticker_data(ticker, cols='date')

    latest_date = existing_data.index.max # TO IMPLEMENT
    # latest_date += 1 day # TO IMPLEMENT

    new_data = get_ticker_history_yfin(ticker, start=latest_date)

    new_data.to_db # TO IMPLEMENT

def pull_ticker_data(ticker: str, cols: str | None = None):
    """
    ticker: str
    cols: str
        Must be in format 'col1, col2, col3'
    """
    if not cols:
        cols = '*'
    with sqlite3.connect('historical_data.db') as con:
        results = con.execute("""
                        SELECT ?
                        FROM historical
                        WHERE ticker = ?
                    """, (cols, ticker))
        results = results.fetchall()
    

def get_ticker_history_yfin(ticker: str, period=None, start=None, end=None):
    """
    ticker: str
        Yahoo Finance stock ticker e.g. ^GSPC for S&P500
    period: str
        None = max, Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    start/end: str
        YYYY-MM-DD, supercedes period
    """
    return yfinance.Ticker(ticker).history(period=period, start=start, end=end, rounding=True, auto_adjust=False)

def create_db() -> None:
    """
    Create an SQLite3 database to store historical market data for selected securities.
    """
    create_hist_table = """
    CREATE TABLE IF NOT EXISTS historical_data (
        ticker_id INTEGER NOT NULL REFERENCES tickers(ticker_id)
            ON DELETE CASCADE
            ON UPDATE CASCADE,
        date TEXT NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        adj_close REAL,
        volume INTEGER,
        dividend REAL DEFAULT 0.0,
        splits REAL DEFAULT 1.0,
        PRIMARY KEY (ticker_id, date) 
        )
    """
    create_ticker_table = """
    CREATE TABLE IF NOT EXISTS tickers (
        ticker TEXT NOT NULL UNIQUE,
        ticker_id INTEGER PRIMARY KEY
        )
    """
    try:
        con = sqlite3.connect('historical_data.db')
        cur = con.cursor()
        cur.execute(create_hist_table)
        cur.execute(create_ticker_table)
        con.commit()
        cur.close()
    except sqlite3.Error:
        print('Unable to execute table creation SQL')
    finally:
        if con:
            con.close()

if __name__ == '__main__':
    main()