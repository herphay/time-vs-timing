#%%
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections.abc import Iterable

from data import pull_ticker_data
from helpers import parse_tickers_n_cols

def main() -> None:
    ...
    # normalize_multi_data(['VT', '^GSPC'], 'adj_close', '1927-12-30')
    missed_n_days(['VT', '^GSPC', '^SP500TR', 'SHY'], [(10,0), (5,5)], 
                  start='2003-01-01', end='2022-12-31', show_n=True)


def summarize_returns(ticker_alloc: dict[str, float],
                      initial_inv: float = 10000) -> pd.DataFrame:
    """
    Summarize the difference in returns for different investment strats across different periods.
    """


def multi_period_missed_n_days(
        tickers: Iterable[str] | str,
        n_scens: Iterable[tuple[int, int]] | tuple[int, int] = ((10, 0),),
        period_start: str | None = None,
        period_end: str | None = None,
        period_len: str = '20y',
        period_freq: str = 'MS',
        initial_inv: float = 10000,
        price_type: str = 'adj_close'
        ) -> pd.DataFrame:
    """
    Summarize the difference in returns across rolling periods if n of the best/worst days are
    missed during the period.
    """
    # 0: Set up basic data
    tickers = parse_tickers_n_cols(tickers)

    if not isinstance(n_scens[0], tuple):
        n_scens = (n_scens,)
    scen_names = [f'Missed {best_n}B, {worst_n}W' for best_n, worst_n in n_scens]

    metrics = ('value_delta', 'value_pct', 'CAGR_delta')

    # 1: get the returns for all relevant tickers & set start/end dates if its None
    returns_df = calc_multi_returns(tickers, price_type, period_start, period_end)

    if not period_start: period_start = returns_df.index[0]
    if not period_end: period_end = returns_df.index[-1]

    # 2: calc all the individual period start end & construct output df
    # 2a: Get the period offset:
    if 'y' in period_len:
        period_len = pd.DateOffset(years=int(period_len.strip('y')))
    elif 'm' in period_len:
        period_len = pd.DateOffset(months=int(period_len.strip('m')))
    elif 'd' in period_len:
        period_len = pd.DateOffset(days=int(period_len.strip('d')))
    else:
        raise ValueError('Period must be an int followed by y, m, or d')

    # 2b: Get the starting dates' range
    starts_s = pd.to_datetime(period_start)            # Start of the starts
    starts_e = pd.to_datetime(period_end) - period_len # End of the starts

    # 2c: Get all starting and ending dates
    starts = pd.date_range(starts_s, starts_e, freq=period_freq)
    ends = pd.DatetimeIndex([s + period_len - pd.DateOffset(days=1) for s in starts])

    # 2d: construct output df
    cols = pd.MultiIndex.from_product([tickers, scen_names, metrics], 
                                      names=['ticker', 'scen', 'metric'])
    periods_data = pd.DataFrame(index=ends, columns=cols)

    # 3: for each period, for each ticker, get the performance data
    for start, end in zip(starts, ends):
        # Slice returns_df to the current period
        period_returns = returns_df.loc[start:end]

        # Remove ticker from calculation if said ticker have missing data in the period
        # If there are any NAs along the column, it will be excluded
        full_data_cols = period_returns.columns[pd.notna(period_returns).all()]
        period_returns = period_returns[full_data_cols]
        period_tickers = period_returns.columns.str.split(' ').str[0]

        # get the results, expected all to be calculated with the full data
        results = missed_n_days(period_tickers, 
                                n_scens, 
                                returns_df=period_returns,
                                start=start,
                                end=end,
                                show_results=False,
                                show_warning=False)
        
        # Now calc the 3 desired metrics ('value_delta', 'value_pct', 'CAGR_dalta')
        for scen in scen_names:
            # Get absolute difference for Final Val & CAGR %
            abs_delta = results.loc[scen] - results.loc['Original Returns']
            abs_delta.index = abs_delta.index.str.replace('Final Value', 'value_delta') \
                                             .str.replace('CAGR %', 'CAGR_delta')
            # Get ratio of scen val to origin val
            rel_delta = (results.loc[scen] / results.loc['Original Returns'] - 1) * 100
            rel_delta.index = rel_delta.index.str.replace('Final Value', 'value_pct')

            deltas = pd.concat([abs_delta, rel_delta])

            #### It is expected that value_delta & CAGR_delta to change for each period.
            #### This is because each period the overall growth % changes due to different
            #### daily returns in the removed & added month.
            #### However, value_pct can remain constant for long stretch of rolling periods.
            #### This is due to the best/worst days being clustered together and when they 
            #### are in the middle of a period, the overall growth effect of their removal
            #### remains constant for all periods around that middle period.
            
            for ticker in period_tickers:
                for metric in metrics:
                    periods_data.loc[end, (ticker, scen, metric)] = deltas[f'{ticker} {metric}']

    # 4: summarize the data for all periods
    return periods_data


def missed_n_days(tickers: Iterable[str] | str,
                  n_scens: Iterable[tuple[int, int]] | tuple[int, int] = ((10, 0),),
                  returns_df: pd.DataFrame | None = None,
                  initial_inv: float = 10000,
                  price_type: str = 'adj_close',
                  start: str | None = None,
                  end: str | None = None,
                  show_n: bool = False,
                  show_results: bool = True,
                  show_warning: bool =True) -> pd.DataFrame:
    """
    Calculate the final value & returns if n of the best/worst days are missed during a period.

    tickers: Iterable | str
        tickers of which returns are to be considered 
    n_scens: Iterable | tuple
        tuple of ints where 
            1st int is the best n days to remove    
            2nd int is worst n days
        All provided tuples will be evaluated
    returns_df: pd.DataFrame | None
        The daily returns DataFrame to be used to calc returns. Note: not the scaling ratio
    """
    # Process tickers -> required for final column re-ordering, unable to rely on sub-funcs
    tickers = parse_tickers_n_cols(tickers)

    # Make returns into absolute multiplier
    if returns_df is not None:
        returns_df = (returns_df + 1)[start:end]
    else:
        returns_df = calc_multi_returns(tickers, price_type, start, end) + 1
    
    returns_df: pd.DataFrame # Explicitly typing the returns_df variable
    
    returns_df.columns = [ticker + ' Final Value' for ticker in tickers]
    
    # .prod() returns a pd.Series with the original df's columns as the index
    original_returns = returns_df.prod() * initial_inv

    # When creating df with dict, the keys become the cols, 
    # the values (if its a pd.Series) then become the row data with series idx as df idx
    # Transposing return the idx (which are original df's cols) into the cols
    compare_df = pd.DataFrame({'Original Returns': original_returns}).T

    # Assuming Period is valid (i.e. there is data from start to end dates) 
    # Get period day count by assuming full period (inclu non-business days) 
    # +1 to the days as dates are inclusive
    ndays = (pd.to_datetime(end) - pd.to_datetime(start)).days + 1
    # If day count pulled deviates more than 10 days (Typically there are only 3 days non-trading)
    # Then update the day count to the available data period
    if ndays > (returns_df.index[-1] - returns_df.index[0]).days + 10:
        ndays = (returns_df.index[-1] - returns_df.index[0]).days + 1
        start = returns_df.index[0].strftime('%Y-%m-%d')

    durations = {}
    if show_results: print('\n\n')

    if not isinstance(n_scens[0], tuple):
        n_scens = (n_scens,)
    
    # For each scenario, calc the returns and append it to compare_df
    for ticker, col in zip(tickers, returns_df.columns):
        cagr_col = ticker + ' CAGR %'

        # For tickers where start/end date are truncated to just the ticker's available date
        if (nadays := pd.isna(returns_df[col]).sum()) > 1 and show_warning:
            print(f'WARNING: {ticker} has {nadays} dates where there are no returns, ' +
                  f"it's final value & CAGR cannot be properly compared to other tickers here.")
            # If there are multiple NaNs, deduct duration til the first valid data point from ndays
            col_days = ndays - ((~pd.isna(returns_df[col])).idxmax() - pd.to_datetime(start)).days - 1
        else:
            col_days = ndays
        
        inv_yrs = 365.25 / col_days
        durations[ticker] = round(1 / inv_yrs, 1)

        compare_df.loc['Original Returns', cagr_col] = ((compare_df.loc['Original Returns', col] / 
                                                         initial_inv) ** inv_yrs - 1) * 100

        if show_n: print('\n', '@' * 80, '\n', '@' * 80, '\n', sep='')
        for scen in n_scens:
            best_n = scen[0]
            worst_n = scen[1]
            best = returns_df.loc[returns_df.nlargest(best_n, col).index, col]
            worst = returns_df.loc[returns_df.nsmallest(worst_n, col).index, col]
            best_total = best.prod()
            worst_total = worst.prod()
            
            if show_n:
                print(f'Missing the best {best_n} & worst {worst_n} days for {col}')
                print('#' * 80, '\n')
                print(f'The best {best_n} daily returns are: ' +
                      f'{' | '.join((((best - 1) * 100).round(1).astype(str) + '%').tolist())}')
                print(f'Cumulatively this is {round((best_total - 1) * 100, 1)}%\n')
                print(f'The worst {worst_n} daily returns are: ' +
                      f'{' | '.join((((worst - 1) * 100).round(1).astype(str) + '%').tolist())}')
                print(f'Cumulatively this is {round((worst_total - 1) * 100, 1)}%\n')
            final_val = original_returns.loc[col] / best_total / worst_total
            compare_df.loc[f'Missed {best_n}B, {worst_n}W', col] = final_val
            compare_df.loc[f'Missed {best_n}B, {worst_n}W', 
                           cagr_col] = ((final_val / initial_inv) ** inv_yrs - 1) * 100
            # modifiers[col] = original_returns.loc[col] / best_total / worst_total
        
        # compare_df.loc[f'Missed {best_n}B, {worst_n}W'] = modifiers
    
    # pd.concat([compare_df, ((compare_df / initial_inv) ** (365.25 / ndays))], axis=1)

    # Rearrange columns
    compare_df = compare_df[[f'{ticker} {val}' for ticker in tickers 
                                               for val in ('Final Value', 'CAGR %')]]

    if show_results:
        print('\n', '@' * 80, '\n', '@' * 80, '\n'*2, sep='')
        print('Years of data available:')
        print(' | '.join([f'{k}: {v} years' for k, v in durations.items()]))
        print(compare_df.round(2))
    
    return compare_df


def calc_multi_returns(tickers: Iterable[str] | str,
                       price_type: str = 'adj_close',
                       start: str | None = None,
                       end: str | None = None,
                       output_format: str = 'df') -> pd.DataFrame | \
                                                dict[str, dict[str, np.ndarray]]:
    """
    Calculate each ticker's daily returns

    tickers: Iterable | str
        Pass 1 or more tickers whose data is to be normalized
        If 'all' is passed, will process data for all available tickers
    price_type: str
        Price type used calculate returns. E.g. close / adj_close
    start/end: str
        ISO 8601 (YYYY-MM-DD) dates to pull data from [start, end]. Default to earliest/latest
    output_format: str 
        Use 'df' to return a DataFrame. Anything else defaults to a nested dict with np.ndarray
    """
    # Pull data earlier than user's start date so we can calc returns for the user's start date
    if start:
        new_start = (pd.to_datetime(start, format='%Y-%m-%d') - timedelta(days=7)).strftime('%Y-%m-%d')
    else:
        new_start = start
    data = process_ticker_data(tickers, price_type, new_start, end)

    if output_format == 'df':
        data = data_df_constructor(data).pct_change()
        data.columns = data.columns.str.replace(price_type, 'daily returns')
        return data.loc[start:end] # Need to use .copy() if we don't want the full df to persist
    else:
        # Get the index location where user start date would be, used for earlier dates removal
        idx = (next(iter(data.values()))['date'] < pd.to_datetime(start)).sum()
        for pdata in data.values():
            # returns the data at the popped key, remove said key/val from dict
            prices = pdata.pop(price_type)
            # calculate returns for 2nd date in the time series onwards 
            # (1st day don't have an earlier date to calc returns with)
            # indexed to idx - 1 as our data now start from the 2nd date onwards
            pdata['daily returns'] = (prices[1:] / prices[:-1] - 1)[idx - 1:]
            # date series have no such truncation as the returns series, so idx stays the same
            pdata['date'] = pdata['date'][idx:]
        return data


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
        ISO 8601 (YYYY-MM-DD) dates to pull data from [start, end]. Default to earliest/latest
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
                    raise ValueError(f'No valid date from {next_valid_idx.strftime('%Y-%m-%d')} ' + 
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
        dates in ISO 8601 (YYYY-MM-DD) format to pull data from [start, end]. Default to earliest/latest
    autodate: bool
        convert datetime to numpy datetime64[ms] for matplotlib auto plot setting
    """
    # Get all available tickers from db if 'all' is passed, else ensure single ticker is iterable
    tickers = parse_tickers_n_cols(tickers)
    # make cols iterable if only a string is passed
    cols = parse_tickers_n_cols(cols)
    
    # Ensure that date is within the data pulled and not duplicated
    cols = ['date'] + [col for col in cols if col != 'date']

    if len(cols) == 1:
        raise ValueError("Can't pull only date data for a ticker, you must specify 1 more column")

    tickers_data = {}
    for ticker in tickers:
        raw_data = pull_ticker_data(ticker, ', '.join(cols), start=start, end=end)
        if not raw_data:
            print(f'No data fetched for ticket {ticker}')
            continue
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


#%%
if __name__ == '__main__':
    main()