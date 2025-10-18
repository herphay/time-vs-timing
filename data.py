import yfinance
from dataclasses import dataclass
from datetime import datetime, timedelta
import sqlite3
import pandas as pd
import time

from tickerinfo import TickerInfo, get_index_list

def main() -> None:
    daily_scrapper()


def daily_scrapper() -> None:
    """
    Function to scrape and append all the new daily price data of 
    a pre-defined list of securities that act as benchmarks for 
    certain asset class.
    """
    benchmark_tickers = [ticker.ticker for ticker in get_index_list()]

    for ticker in benchmark_tickers:
        if ticker_scrapper(ticker):
            time.sleep(2.5)
    print(f'\n########## Database updated for all {len(benchmark_tickers)} ' +
           'benchmark indices ##########')


def ticker_scrapper(ticker: str) -> bool:
    """
    Scrape Yahoo Finance for the historical data for a specific ticker.
    Only data that has not yet been scrapped before (not in database) will be added.
    """
    today = datetime.today().strftime('%Y-%m-%d')
    latest_date = pull_ticker_data(ticker, cols='MAX(date)')
    latest_date = (datetime.strptime(latest_date[0][0], '%Y-%m-%d') + 
                   timedelta(days=1)).strftime('%Y-%m-%d') if \
                  len(latest_date) == 1 else None
    if latest_date >= today:
        print(f'{ticker} data has already been updated til {today}, no need to update')
        return False

    new_data = get_ticker_history_yfin(ticker, start=latest_date)
    new_data = new_data.reset_index()
    new_data['Date'] = new_data['Date'].dt.strftime('%Y-%m-%d')
    time_range = f'{'earliest' if not latest_date else latest_date} to {today} (not incl.)'

    if not (ticker_id := pull_ticker_id(ticker)):
        add_ticker(ticker)
        ticker_id = pull_ticker_id(ticker)

    # Add in ticker_id
    new_data['ticker_id'] = ticker_id

    # ensure all required columns are persent and in the right order
    new_data = new_data[['ticker_id', 'Date', 'Open', 'High',  
                         'Low', 'Close', 'Adj Close', 'Volume',
                          'Dividends', 'Stock Splits']]
    
    # print(list(new_data.itertuples(index=False, name=None)))

    push_ticker_data(list(new_data.itertuples(index=False, name=None)))
    print(f'Successfully updated database with {time_range} daily price data for ticker: {ticker}')
    return True


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
    print(f'ticker {ticker} added, please update the other attributes later')


def pull_ticker_id(ticker: str) -> int | None:
    with sqlite3.connect('historical_data.db') as con:
        result = con.execute("SELECT ticker_id FROM tickers WHERE ticker = ?", 
                             (ticker,)).fetchone()
        return result[0] if result is not None else None


def pull_ticker_data(
        ticker: str, 
        cols: str | None = None,
        start: str | None = None,
        end: str | None = None
    ) -> list[tuple]:
    """
    Pull historical data from the local database. Returns as rows.
    
    ticker: str    
    cols: str
        Must be in format 'col1, col2, col3'    
    start/end: str
        ISO 8601 (YYYY-MM-DD) format. Selecting [start:end]
        Defaults to earliest/latest dates if None
    """
    if not cols: cols = '*'
    if not start: start = '0'
    if not end: end = '9'
    with sqlite3.connect('historical_data.db') as con:
        results = con.execute(f" \
                        SELECT {cols} \
                        FROM historical_data \
                        WHERE ticker_id = (SELECT ticker_id \
                                           FROM tickers \
                                           WHERE ticker = ?) \
                        AND date >= ? \
                        AND date <= ? \
                    ", (ticker, start, end))
        return results.fetchall()
    

def get_ticker_history_yfin(
        ticker: str, 
        period=None, 
        start=None, 
        end=None
    ) -> pd.DataFrame:
    """
    ticker: str
        Yahoo Finance stock ticker e.g. ^GSPC for S&P500
    period: str
        None = max, Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    start/end: str
        YYYY-MM-DD. Supercedes period. None defaults to earliest/latest date.
    """
    return yfinance.Ticker(ticker).history(period=period, start=start, end=end, rounding=True, auto_adjust=False)


def db_initialize() -> None:
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
        ticker_id INTEGER PRIMARY KEY,
        ticker TEXT NOT NULL UNIQUE,
        asset_class TEXT,
        sub_class TEXT,
        benchmark TEXT,
        remarks TEXT
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

    # After database creation, load in the initial tickers
    update_db_tickers(get_index_list())


def update_db_tickers(tickers: list[TickerInfo]) -> None:
    ticker_details = [
        (ticker.ticker, ticker.Asset_Class, ticker.Sub_Class,
         ticker.Benchmark, ticker.Remarks)
        for ticker in tickers
    ]

    with sqlite3.connect('historical_data.db') as con:
        con.executemany("""
            INSERT INTO tickers (ticker, asset_class, 
                                 sub_class, benchmark, 
                                 remarks)
                        VALUES (?, ?, ?, ?, ?)
        """, ticker_details)


def get_all_tickers() -> list[str]:
    """Return list of all tickers currently available in the local database"""
    with sqlite3.connect('historical_data.db') as con:
        results = con.execute('SELECT ticker FROM tickers').fetchall()
        return [tup[0] for tup in results]
    

def get_all_start_dates(
        to_print: bool = True,
        get_results: bool = False
    ) -> dict[str, str]:
    """Return a list of tuples with (ticker, earliest data available date)"""
    with sqlite3.connect('historical_data.db') as con:
        results =  con.execute("""SELECT ticker, MIN(date) FROM 
                                  tickers t JOIN
                                  historical_data h
                                  ON t.ticker_id = h.ticker_id
                                  GROUP BY ticker
                                  ORDER BY MIN(date)
                               """).fetchall()
        if to_print:
            results = [('Ticker', 'Earliest data')] + results
            for tup in results:
                print(f'{tup[0]:17}: {tup[1]}')
        
        if get_results:
            return dict(results)


if __name__ == '__main__':
    main()