import numpy as np
import pandas as pd

from data import pull_ticker_data

def main() -> None:
    ...


def process_ticker_data(ticker: str, 
                        cols: tuple[str], 
                        start: str | None = None, 
                        end: str | None = None,
                        autodate: bool = True) -> dict[str, np.ndarray]:
    """
    Pull the required ticker data & process it into a dict of Numpy Arrays for each column

    ticker: str     
        Ticker which data is to be pulled
    cols: tuple     
        tuple of column names of which data is to be pulled
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