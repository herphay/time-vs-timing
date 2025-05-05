import numpy as np
import pandas as pd
from collections.abc import Iterable

from data import pull_ticker_data, get_all_tickers

def main() -> None:
    ...
    # normalize_multi_data(['VT', '^GSPC'], 'adj_close', '1927-12-30')


def calc_multi_returns(ticker: Iterable[str] | str,
                       col: str = 'adj_close',
                       start: str | None = None,
                       end: str | None = None,
                       out_type: str = '') -> dict[str, dict[str, np.ndarray]] | \
                                              pd.DataFrame:
    ...

def normalize_multi_data(tickers: Iterable[str] | str,
                         data_col: str,
                         ref_date: str,
                         truncate: bool = False,
                         same_ref: bool = True,
                         start: str | None = None,
                         end: str | None = None) -> pd.DataFrame:
    """
    Normalize multiple tickers' selected data where a specific date = 100%

    tickers: Iterable | str
        Pass 1 or more tickers whose data is to be normalized
        If 'all' is passed, will process data for all available tickers
    data_col: str
        Column name whose data is to be normalized. E.g. close
    ref_date: str
        ISO 8601 (YYYY-MM-DD) reference date where data is set to be 100%
    truncate: bool
        True:  Only keep dates where all tickers' have data
        False: All tickers' full data will be kept
    same_ref: bool
        True:  All tickers must have the same reference -> from the ref_date, find the earliest
               ref_date that allows all tickers to have the same ref without error (notna, not 0)
        False: Not all tickers must have the same reference -> Each ticker to find the earliest
               ref_date that will give that ticker a ref without error (notna, not 0)
    start/end: str
        ISO 8601 (YYYY-MM-DD) dates to pull data from [start, end). Default to earliest/latest
    """
    # Get the data processed into the form of a pd.DataFrame
    data_df = data_df_constructor(process_ticker_data(tickers, data_col, start, end), truncate)

    # Now to rebase the time series
    # ref_date = pd.to_datetime(ref_date, format='%Y-%m-%d') # Not needed, pd.loc for dtIndex work on str
    try:
        base_val = data_df.loc[ref_date].copy() # As base_val might be changed later on, copy it
        forward_df = data_df.loc[ref_date:]
    except KeyError:
        # Get the next closest date to be used as the reference instead
        new_ref = data_df.index[data_df.index.get_indexer([pd.to_datetime(ref_date)], 
                                                          method='bfill')]
        new_ref = new_ref.strftime('%Y-%m-%d')[0]
        print(f'Input reference date {ref_date} does not exist in the time series. Using ' + 
              f'next closest date {new_ref} instead')
        
        base_val = data_df.loc[new_ref].copy() # As base_val might be changed later on, copy it
        forward_df = data_df.loc[new_ref:]
    
    # If all tickers must have the same reference date
    if same_ref:
        # Check if user supplied reference date has any invalid data
        col_has_error = (base_val.isna() | (base_val == 0)) # pd.Series of bool with idx of col name
        error_cols = base_val[col_has_error].index # Get the col names where has error is true

        if col_has_error.any():
            # Check the entire future df, get the first row idx where all tickers are valid
            # If no rows are valid, idxmax will return the original row index
            next_valid_idx = (forward_df.notna() & (forward_df != 0)).all(axis=1).idxmax()

            # If the new idx is the original idx, it must mean there are no valid future rows
            if next_valid_idx == forward_df.index[0]:
                    raise ValueError(f'No valid date from {next_valid_idx.strftune('%Y-%m-%d')} ' + 
                                     'onwards where all requested tickers have valid data')
            
            # Otherwise there are valid rows and we update to that row's data
            base_val = forward_df.loc[next_valid_idx].copy()
            print(f'Original reference date {ref_date} contains invalid data for: ' + 
                  f'{', '.join(error_cols)}. Update normalization reference to next ' + 
                  f'valid date {next_valid_idx.strftime('%Y-%m-%d')}.')
            
    # Otherwise each ticker data can have separate reference dates
    else:
        # For each ticker data
        for col, val in base_val.items():
            # If the data is not errorneous, continue
            if pd.notna(val) and val != 0:
                continue

            # Else get the next valid index
            next_valid_idx = (forward_df[col].notna() & (forward_df[col] != 0)).idxmax(axis=0)

            # If next valid index is the original index, it must be the case where there are no 
            # valid data from the original ref date onwards, raise error
            if next_valid_idx == forward_df.index[0]:
                raise ValueError(f'No valid value for {col} from ' +
                                f'{next_valid_idx.strftime('%Y-%m-%d')} onwards')
            # Update this ticker data's reference point to the next valid date
            base_val[col] = forward_df.loc[next_valid_idx, col]
            print(f'Original reference date {ref_date} contains invalid data for: ' +
                  f'{col}. Updating the reference data for it to new reference date ' +
                  f'{next_valid_idx.strftime('%Y-%m-%d')}')

    # Once all reference values are updated properly, normalize the data
    data_df = data_df / base_val * 100

    return data_df


def data_df_constructor(ticker_dict: dict[str, dict[str, np.ndarray]],
                        truncate: bool = False) -> pd.DataFrame:
    """
    Takes in a ticker data dict in the specified form and transform it into a DataFrame

    ticker_dict: dict[str, dict[str, np.ndarray]]
        Dict of dicts, 1st level associate tickers to it's data, 2nd level associate col_name to data
    truncate: bool
        True:  Only keep dates where all tickers' have data
        False: All tickers' full data will be kept
    """
    # Update data to a pd.DataFrame rather than a dict & rename them appropriately
    data = {ticker: pd.DataFrame(data_dict).set_index('date') for ticker, data_dict in ticker_dict.items()}
    # for ticker, df in data.items():
    #     df.columns = ticker + '_' + df.columns

    method = 'inner' if truncate else 'outer'

    # When passing dict[str: df] to pd.concat(), keys will auto set to the dict keys str
    merged_df = pd.concat(data, axis=1, join=method, sort=True)
    # map will map the enclosed function to each of the output, in this case tuple of multiIndex col names
    merged_df.columns = merged_df.columns.map(' '.join)

    return merged_df


def process_ticker_data(tickers: Iterable[str] | str, 
                        cols: Iterable[str] | str, 
                        start: str | None = None, 
                        end: str | None = None,
                        autodate: bool = True) -> dict[str, dict[str, np.ndarray]]:
    """
    Pull the required ticker data & process it into a dict of Numpy Arrays for each column
    which is then stored in a dict of tickers

    tickers: Iterable[str] | str    
        Ticker(s) which data is to be pulled
    cols: Iterable[str] | str
        Column name(s) of which data is to be pulled. Date + 1 additional col is necessarily pulled.
    start/end: str
        dates in ISO 8601 (YYYY-MM-DD) format to pull data from [start, end). Default to earliest/latest
    autodate: bool
        convert datetime to numpy datetime64[ms] for matplotlib auto plot setting
    """
    # Get all available tickers from db if 'all' is passed, else ensure single ticker is iterable
    if isinstance(tickers, str):
        if tickers == 'all':
            tickers = get_all_tickers()
        else:
            tickers = (tickers,)
    # make cols iterable if only a string is passed
    if isinstance(cols, str):
        cols = (cols,)
    
    # Ensure that date is within the data pulled and not duplicated
    cols = ['date'] + [col for col in cols if col != 'date']

    if len(cols) == 1:
        raise ValueError("Can't pull only date data for a ticker, you must specify 1 more column")

    tickers_data = {}
    for ticker in tickers:
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
        
        # Get the data into {ticker: data_dict} format
        tickers_data[ticker] = packed_data

    return tickers_data


if __name__ == '__main__':
    main()