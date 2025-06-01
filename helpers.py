import numpy as np
import pandas as pd
from scipy.optimize import newton
from collections.abc import Iterable

from data import get_all_tickers, pull_ticker_data

def parse_tickers_n_cols(
        items: Iterable[str] | str
    ) -> Iterable[str]:
    # Get all available tickers from db if 'all' is passed ('all' is only used by tickers)
    # else ensure single ticker/column is iterable
    if isinstance(items, str):
        if items == 'all':
            items = get_all_tickers()
        else:
            items = (items,)
    return items

def xirr(
        cashflows: pd.Series,
        rate_guess: float = 0.1,
        days_in_yr: float = 365
    ) -> float:
    """
    Calculates the XIRR for a series of casflows. By convention -ve are cash invested, +ve are 
    withdrawals.

    cashflows: pd.Series
        A pandas series with datetime index and corresponding cashflow as the values
    """
    cashflows = cashflows.sort_index()
    first_date = cashflows.index[0]

    # Get the powers for the algebriac XNPV function
    powers = np.array([(date - first_date).days / days_in_yr for date in cashflows.index])

    # Get the coefficients for the algebriac XNPV function
    # Not needed, because the cashflows are the coefficients
    cashflows = np.array(cashflows)

    # Get the powers for the derivative of the XNPV function
    deri_powers = powers - 1

    # Get the coefficients for the derivative of the XNPV function
    deri_coeff = cashflows * powers

    # call scipy.optimize.newton and pass the XNPV func along with its 1st derivative
    # Here, solve for x where x = 1 / (1 + xirr)
    inv_r = newton(
                func=lambda x: np.sum(cashflows * x ** powers),
                x0 = 1 / (1 + rate_guess),
                fprime=lambda x: np.sum(deri_coeff * x ** deri_powers)
            )

    return 1 / inv_r - 1


"""
s0 = pd.Series([-1000, -1000, -1000, 1100, 5000], index=pd.to_datetime(['1/1/2020', '1/2/2021', '3/5/2022', '9/22/2023', '5/1/2025'], format='%m/%d/%Y'))
# 0.1932151
s1 = pd.Series([-100, 11, 121], index=pd.to_datetime(['1/1/2021', '1/1/2023', '1/1/2024'], format='%m/%d/%Y'))
# 0.1
s2 = pd.Series([-1000, 200, 250, 300, 350], index=pd.to_datetime(['2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01', '2024-01-01'], format='%Y-%m-%d'))
# 0.0358109
s3 = pd.Series([-5000, 1000, 3000], index=pd.to_datetime(['2020-01-15', '2021-03-10', '2022-07-01'], format='%Y-%m-%d'))
# -0.0986186
s4 = pd.Series([-1000, 250, 250, 505], index=pd.to_datetime(['2021-06-01', '2022-06-01', '2023-06-01', '2024-06-01'], format='%Y-%m-%d'))
# 0.00221486
s5 = pd.Series([-10000, 10200], index=pd.to_datetime(['2024-01-01', '2024-04-01'], format='%Y-%m-%d'))
# 0.08266773
s6 = pd.Series([-100, 0, -50, 0, 180], index=pd.to_datetime(['2020-01-01', '2020-07-01', '2021-01-01', '2021-07-01', '2022-01-01'], format='%Y-%m-%d'))
# 0.11459908
"""


def parse_prices_for_inv_style(
        ticker: str,
        start: str,
        end: str,
        prices: pd.DataFrame | None = None,
        price_type: str = 'adj_close'
    ) -> pd.Series | pd.DataFrame:
    """
    Help investment style funcs check and return a price pd.Series for a specific ticker
    """
    # ticker = parse_tickers_n_cols(ticker)[0] # Not needed as we only expect a singular ticker
    # To squeeze df into pd.Series to avoid needing to access df col in the purchase price loop
    # Saved ~0.6ms per run for 53 month loop run
    if prices is None:
        prices = data_df_constructor(
            process_ticker_data(
                ticker,
                cols=price_type,
                start=start,
                end=end
            )
        ).squeeze()
    else:
        prices = prices.loc[start:end].squeeze()
    
    if (pd.to_datetime(start) - prices.index[0]).days < -4 or \
       (pd.to_datetime(end) - prices.index[-1]).days > 4:
        raise ValueError('start or end date is out of range of available price data')
    
    prices: pd.Series | pd.DataFrame

    if isinstance(prices, pd.DataFrame):
        prices.columns = ticker
    
    return prices


def data_df_constructor(
        ticker_dict: dict[str, dict[str, np.ndarray]],
        truncate: bool = False
    ) -> pd.DataFrame:
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


def process_ticker_data(
        tickers: Iterable[str] | str, 
        cols: Iterable[str] | str, 
        start: str | None = None, 
        end: str | None = None,
        autodate: bool = True
    ) -> dict[str, dict[str, np.ndarray]]:
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