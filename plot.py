import matplotlib.pyplot as plt
import numpy as np
from collections.abc import Iterator
from data import pull_ticker_data

def main() -> None:
    plot_day_range('VT', start='2025-01-01')
    plot_composite('VT', start='2025-01-01')
    plt.show() # plt.show() only outside to ensure all plots can show together


def plot_day_range(ticker: str, start: str = None, end: str = None) -> None:
    """
    ticker: str
    start/end: str
        ISO 8601 (YYYY-MM-DD)
    """
    bins = 21

    # Obtain historical data used in plotting
    cols = ('date', 'high', 'low')
    raw_data = pull_ticker_data(ticker, cols=', '.join(cols), start=start, end=end)
    # zip(*raw_data) converts list of tuples (of rows) into list of tuples (of columns)
    data_dict = numpyfy_data(zip(cols, zip(*raw_data)))

    fig, ax = plt.subplots()
    ax.fill_between(data_dict['date'], data_dict['high'], data_dict['low'], alpha=.5, linewidth=0)
    # ax.plot(dates, (highs + lows)/2, linewidth=2)

    ### Manual tick location setting ###
    # Manual calc (round to int)
    xidx = get_date_idx(len(data_dict['date']), bins)
    
    # Using Numpy linspace (round down to int)
    # x_idx2 = np.array(np.linspace(0, len(dates)-1, 21, dtype=int))
    
    ax.set_xticks(xidx, data_dict['date'][xidx], rotation=45, ha='right')
    # ax.tick_params('x', rotation=90)   # redundant, another way to set rotation
    # fig.autofmt_xdate()                # Auto rotate dates but does not properly increase tick gap

    set_ax_labels(ax, f'Daily price range against time for: {ticker}')

    # plt.show(block=False)


def plot_composite(ticker: str, start: str = None, end: str = None) -> None:
    """
    ticker: str
    start/end: str
        ISO 8601 (YYYY-MM-DD)
    """
    bins = 21

    cols = ('date', 'high', 'low', 'open', 'close')
    raw_data = pull_ticker_data(ticker, cols=', '.join(cols), start=start, end=end)
    data_dict = numpyfy_data(zip(cols, zip(*raw_data)))
    # Combine open and close data into 1 2-dim array
    # transpose to make it Nx2 array where N is num of dates
    # Essentially make each row correspond to a date, and 1st col is open, 2nd is close
    opclo  = np.array([data_dict['open'], data_dict['close']]).T

    fig, ax = plt.subplots()

    ax.fill_between(data_dict['date'], data_dict['high'], data_dict['low'], alpha=0.5, linewidth=0)

    # plotting a NxM array in y-axis means plotting M lines with N datapoints each
    ax.plot(data_dict['date'], opclo, label=['open', 'close'])

    xidx = get_date_idx(len(data_dict['date']), bins)

    ax.set_xticks(xidx, data_dict['date'][xidx], rotation=45, ha='right')

    set_ax_labels(ax, f'Daily price range and open/close prices against time for: {ticker}')
    ax.legend() # to show legend for lines

    # plt.show(block=False)


def numpyfy_data(col_datas: Iterator[tuple[str, tuple]]) -> dict:
    """
    col_datas: zip object
        Expects a zip object combining column name with the corresponding time series
    """
    return {col_name: np.array(col_data)
            for col_name, col_data
            in col_datas}

def get_date_idx(len: int, bins: int) -> list:
    bins = min(bins, len)
    return [round(pos * (len - 1) / (bins - 1)) for pos in range(bins - 1)] + [len - 1]


def set_ax_labels(ax: plt.Axes, 
                  title: str, 
                  xlabel: str = 'Dates', 
                  ylabel: str = 'Price') -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


if __name__ == '__main__':
    main()