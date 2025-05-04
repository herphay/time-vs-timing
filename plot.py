import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# from collections.abc import Iterator

from analysis import normalize_multi_data
from analysis import process_ticker_data

# Global data definition
bins = 21 # Number of x-axis ticks to display for charts

def main() -> None:
    plot_day_range('VT', start='2025-01-01')
    plot_composite('VT', start='2025-01-01')
    plot_single('VT', 'close', start='2025-01-01')
    plot_single('VT', 'open', start='2025-01-01', autodate=False)
    plt.show() # plt.show() only outside to ensure all plots can show together


def plot_df(df: pd.DataFrame) -> None:
    ax = setup_plot_elements('Normalize')
    df.plot(ax=ax)

def plot_day_range(ticker: str, 
                   start: str | None = None, 
                   end: str | None = None, 
                   autodate: bool = True) -> None:
    """
    ticker: str
    start/end: str
        ISO 8601 (YYYY-MM-DD)
    autodate: bool
        Whether we use matplotlib default date locator
    """

    # Obtain historical data used in plotting
    cols = ('date', 'high', 'low')
    data = process_ticker_data(ticker, cols, start=start, end=end, autodate=autodate)

    ax = setup_plot_elements(f'Daily price range against time for: {ticker}', 
                             dates=data['date'],
                             autodate=autodate)
    
    ax.fill_between(data['date'], data['high'], data['low'], alpha=.5, linewidth=0)
    # ax.plot(dates, (highs + lows)/2, linewidth=2)

    # plt.show(block=False) # For interactive mode but the moment the local function context ends, plot will also close


def plot_composite(ticker: str, 
                   start: str | None = None, 
                   end: str | None = None, 
                   autodate: bool = True) -> None:
    """
    ticker: str
    start/end: str
        ISO 8601 (YYYY-MM-DD)
    autodate: bool
        Whether we use matplotlib default date locator
    """

    cols = ('date', 'high', 'low', 'open', 'close')
    data = process_ticker_data(ticker, cols, start=start, end=end, autodate=autodate)

    # Combine open and close data into 1 2-dim array
    # transpose to make it Nx2 array where N is num of dates
    # Essentially make each row correspond to a date, and 1st col is open, 2nd is close
    opclo  = np.stack([data['open'], data['close']]).T

    ax = setup_plot_elements(f'Daily price range and open/close prices against time for: {ticker}', 
                             dates=data['date'],
                             autodate=autodate)

    ax.fill_between(data['date'], data['high'], data['low'], alpha=0.5, linewidth=0)

    # plotting a NxM array in y-axis means plotting M lines with N datapoints each
    ax.plot(data['date'], opclo, label=['open', 'close'])

    ax.legend() # to show legend for lines; Must be called after line's legent are defined through label=labels

    # plt.show(block=False)


def plot_single(ticker: str, 
                col: str,
                start: str | None = None, 
                end: str | None = None, 
                autodate: bool = True) -> None:
    """
    ticker: str
    col: str
        The single column to plot
    start/end: str
        ISO 8601 (YYYY-MM-DD)
    autodate: bool
        Whether we use matplotlib default date locator
    """
    cols = ('date', col)
    data = process_ticker_data(ticker, cols=cols, start=start, end=end, autodate=autodate)

    ax = setup_plot_elements(f'Daily closing price for: {ticker}', 
                             dates=data['date'], 
                             autodate=autodate)

    ax.plot(data['date'], data[col], label=[col])
    ax.legend()


##### No longer required after re-factoring all data processing to dedicated process_ticker_data func #####
# def numpyfy_data(col_datas: Iterator[tuple[str, tuple]]) -> dict[str, np.ndarray]:
#     """
#     col_datas: zip object
#         Expects a zip object combining column name with the corresponding time series
#     """
#     return {col_name: np.array(col_data)
#             for col_name, col_data
#             in col_datas}

def get_date_idx(length: int, bins: int) -> list[int]:
    """Calculate evenly spaced index positions for chart plotting"""
    bins = min(bins, length)
    if bins == 1:
        return [0]
    return [round(pos * (length - 1) / (bins - 1)) for pos in range(bins - 1)] + [length - 1]


def setup_plot_elements(title: str,
                        xlabel: str = 'Dates', 
                        ylabel: str = 'Price',
                        dates: np.ndarray | None = None,
                        manual_fmt: bool = True,
                        autodate: bool = True) -> plt.Axes:
    """
    Set up a plot and all basic elements including tick locations

    dates: np.ndarray    
        Numpy array of dates used in the plot   
    title: str    
        Title of the chart  
    autodate: bool
        Determine whether x-axis date tickers are set up by matplotlib auto locator    
        OR
        manual calculation of bins amount of equally spaced ticks
        Both will always include the start & end date
    """
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    manual_fmt = False if dates is None else manual_fmt
    if manual_fmt:
        manual_set_xaxis(ax, fig, dates, autodate)
    
    fig.tight_layout() # To make plots fit nicely in plotted area, need to call fig by fig, not all at once at plt
    return ax


def manual_set_xaxis(ax: plt.Axes,
                     fig: plt.Figure,
                     dates: np.array,
                     autodate: bool) -> None:
    """
    For manual plotting of x-axis where we want a specific manual formatting method
    """
    if autodate and len(dates) > 1:
        # modify start/end date to matplotlib specific dates numbers
        start_date_num = mdates.date2num(dates[0])
        end_date_num   = mdates.date2num(dates[-1])

        # Set x-axis limits to ensure chart fill up the entire axis
        # also critical to help AutoDateLocator to locate proper dates 
        ax.set_xlim(dates[0], dates[-1])

        # Get the auto-calculated tick locations
        # For some reason, calling the tick_values method directly like this require
        # us to convert the input date values as python datetime dtype, else it will be a TypeError
        # Numpy datetime64 objects can be converted to python datetime objects with .item() method
        # However, the datetime64 object must have at least ms precision -> datetime64[ms]
        auto_ticks = mdates.AutoDateLocator().tick_values(dates[0].item(), dates[-1].item())

        # To ensure both the start & end dates are included in the ticks
        # we will join the auto calculated ticks with the start/end date ticks
        # np.unique then return the unique sorted ticks to be set as the x-axis
        all_ticks = np.unique(np.concat((auto_ticks, (start_date_num, end_date_num))))
        ax.set_xticks(all_ticks)
        fig.autofmt_xdate(rotation=45, ha='right')
    else:
        ##### Manual tick location setting #####
        # treat dates as int index positions -> 
        # dates won't be spaced properly as non-trading days are missing
        ax.set_xlim(0, len(dates) - 1)          
        # ax.set_xlim(dates[0], dates[-1]) # This is for treating dates like dates, prereq of dates in datetime dtype
        # Manual calc (round to int)
        xidx = get_date_idx(len(dates), bins)
        
        # Using Numpy linspace (round down to int)
        # x_idx2 = np.array(np.linspace(0, len(dates)-1, 21, dtype=int))

        ax.set_xticks(xidx, dates[xidx], rotation=45, ha='right')
        # Next row is for treating dates like dates, it will having the right date spacing 
        # (date spaced inclusive of non-trading days)
        # But the tickers won't be spaced properly -> cause int spacing ignores non-trading days
        # ax.set_xticks(dates[xidx], pd.Series(dates[xidx]).dt.strftime('%Y-%m-%d'), rotation=45, ha='right')

        # ax.tick_params('x', rotation=90)   # redundant, another way to set rotation
        # fig.autofmt_xdate()                # Auto rotate dates but does not properly increase tick gap


if __name__ == '__main__':
    main()