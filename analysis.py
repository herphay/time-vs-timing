import numpy as np
import pandas as pd
from collections.abc import Iterable

from data import pull_ticker_data

def main() -> None:
    ...


def normalize_multi_data(ticker: Iterable[str] | str,
                         data_col: str,
                         ref_date: str,
                         truncate: bool = False,
                         start: str | None = None,
                         end: str | None = None) -> pd.DataFrame:
    """
    Normalize multiple ticker's selected data where a specific date = 100%

    ticker: Iterable | str
        Pass 1 or more ticker whose data is to be normalized
    data_col: str
        Column name whose data is to be normalized. E.g. close
    ref_date: str
        ISO 8601 (YYYY-MM-DD) reference date where data is set to be 100%
    truncate: bool
        True:  Only keep dates where all tickers' have data
        False: All tickers' full data will be kept
    start/end: str
        ISO 8601 (YYYY-MM-DD) dates to pull data from [start, end). Default to earliest/latest
    """
    # Minimum data required is date + the associated data_col series
    cols = ('date', data_col)

    # Get the data into {ticker: data_dict} format
    if isinstance(ticker, str):
        data = {ticker: process_ticker_data(ticker, cols, start=start, end=end)}
    else:
        data = {tick: process_ticker_data(tick, cols, start=start, end=end) for tick in ticker}
    
    # Update data to a pd.DataFrame rather than a dict
    data = {ticker: pd.DataFrame(data_dict).set_index('date') for ticker, data_dict in data.items()}


def process_ticker_data(ticker: str, 
                        cols: Iterable[str], 
                        start: str | None = None, 
                        end: str | None = None,
                        autodate: bool = True) -> dict[str, np.ndarray]:
    """
    Pull the required ticker data & process it into a dict of Numpy Arrays for each column

    ticker: str     
        Ticker which data is to be pulled
    cols: Iterable     
        Iterable of column names of which data is to be pulled
    start/end: str
        dates in ISO 8601 (YYYY-MM-DD) format to pull data from [start, end). Default to earliest/latest
    autodate: bool
        convert datetime to numpy datetime64[ms] for matplotlib auto plot setting
    """
    raw_data = pull_ticker_data(ticker, ', '.join(cols), start=start, end=end)
    if not raw_data:
        print(f'No data fetched for ticket {ticker}')
        return
    # zip(*raw_data) converts list of tuples (of rows) into list of tuples (of columns)
    raw_data = zip(*raw_data)         # Convert row data to col data
    packed_data = zip(cols, raw_data) # Attach col name to each col data

    packed_data = {col_name: np.array(col_data) for col_name, col_data in packed_data}
    if autodate:
        # specify Numpy datetime64 dtype, with millisecond precision [ms] rather than day [D]
        # Reason being we need ms precision for the datetime64 object to be successfully converted
        # to standard python datetime object later in setup_plot_elements section with .item() method 
        packed_data['date'] = np.array(packed_data['date'], dtype='datetime64[ms]')
    return packed_data


if __name__ == '__main__':
    main()