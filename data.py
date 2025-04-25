import yfinance
from datetime import datetime
import sqlite3
import pandas as pd

def main() -> None:
    ...


def ticker_scrapper(ticker: str) -> None:
    """
    Scrape Yahoo Finance for the historical data for a specific ticker.
    Only data that has not yet been scrapped before (not in database) will be added.
    """
    # today = datetime.today().strftime('%Y-%m-%d')
    latest_date = pull_ticker_data(ticker, cols='MAX(date)')
    latest_date = latest_date[0][0] if len(latest_date) == 1 else None

    new_data = get_ticker_history_yfin(ticker, start=latest_date)
    new_data = new_data.reset_index()
    new_data['Date'] = new_data['Date'].dt.strftime('%Y-%m-%d')

    if not (ticker_id := pull_ticker_id(ticker)):
        add_ticker(ticker)
        ticker_id = pull_ticker_id(ticker)

    # Add in ticker_id
    new_data['ticker_id'] = ticker_id

    # ensure all required columns are persent and in the right order
    new_data = new_data[['ticker_id', 'Date', 'Open', 'High',  
                         'Low', 'Close', 'Adj Close', 'Volume',
                          'Dividends', 'Stock Splits']]
    
    print(list(new_data.itertuples(index=False, name=None)))

    push_ticker_data(list(new_data.itertuples(index=False, name=None)))


def push_ticker_data(data: list[tuple]) -> None:
    insertion_sql = """
    INSERT INTO historical_data (ticker_id, date, open, high, low, close,
                                 adj_close, volume, dividend, splits)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    con = sqlite3.connect('historical_data.db')
    try:
        with con:
            con.executemany(insertion_sql, data)
    except sqlite3.Error:
        print('Data insert error')
    finally:
        con.close()


def add_ticker(ticker: str) -> None:
    with sqlite3.connect('historical_data.db') as con:
        con.execute("INSERT INTO tickers (ticker) VALUES (?)", (ticker,))


def pull_ticker_id(ticker: str) -> int | None:
    with sqlite3.connect('historical_data.db') as con:
        result = con.execute("SELECT ticker_id FROM tickers WHERE ticker = ?", 
                             (ticker,)).fetchone()
        return result[0] if result is not None else None


def pull_ticker_data(ticker: str, cols: str | None = None) -> list:
    """
    ticker: str
    cols: str
        Must be in format 'col1, col2, col3'
    """
    if not cols:
        cols = '*'
    with sqlite3.connect('historical_data.db') as con:
        results = con.execute(f" \
                        SELECT {cols} \
                        FROM historical_data \
                        WHERE ticker_id = (SELECT ticker_id \
                                           FROM tickers \
                                           WHERE ticker = ?) \
                    ", (ticker,))
        return results.fetchall()
    

def get_ticker_history_yfin(ticker: str, period=None, start=None, end=None) -> pd.DataFrame:
    """
    ticker: str
        Yahoo Finance stock ticker e.g. ^GSPC for S&P500
    period: str
        None = max, Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    start/end: str
        YYYY-MM-DD. Supercedes period. None defaults to earliest/latest date.
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