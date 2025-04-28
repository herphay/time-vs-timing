import matplotlib.pyplot as plt
import numpy as np
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
    data = pull_ticker_data(ticker, cols='date, high, low', start=start, end=end)
    dates, highs, lows = zip(*data) # Convert list of tuples (of rows) into list of tuples (of columns)
    dates = np.array(dates)
    highs = np.array(highs)
    lows = np.array(lows)

    fig, ax = plt.subplots()
    ax.fill_between(dates, highs, lows, alpha=.5, linewidth=0)
    # ax.plot(dates, (highs + lows)/2, linewidth=2)

    ### Manual tick location setting ###
    # Manual calc (round to int)
    xidx = get_date_idx(len(dates), bins)
    
    # Using Numpy linspace (round down to int)
    # x_idx2 = np.array(np.linspace(0, len(dates)-1, 21, dtype=int))
    
    ax.set_xticks(xidx, dates[xidx], rotation=45, ha='right')
    # ax.tick_params('x', rotation=90)   # redundant, another way to set rotation
    # fig.autofmt_xdate()                # Auto rotate dates but does not properly increase tick gap

    ax.set_title(f'Daily price range against time for: {ticker}')
    ax.set_xlabel('Dates')
    ax.set_ylabel('Price')

    # plt.show(block=False)


def plot_composite(ticker: str, start: str = None, end: str = None) -> None:
    """
    ticker: str
    start/end: str
        ISO 8601 (YYYY-MM-DD)
    """
    bins = 21

    data = pull_ticker_data(ticker, cols='date, high, low, open, close', start=start, end=end)
    dates, highs, lows, opens, closes = zip(*data)
    dates  = np.array(dates)
    highs  = np.array(highs)
    lows   = np.array(lows)
    # Combine open and close data into 1 2-dim array
    # transpose to make it Nx2 array where N is num of dates
    # Essentially make each row correspond to a date, and 1st col is open, 2nd is close
    opclo  = np.array([opens, closes]).T

    fig, ax = plt.subplots()

    ax.fill_between(dates, highs, lows, alpha=0.5, linewidth=0)

    # plotting a NxM array in y-axis means plotting M lines with N datapoints each
    ax.plot(dates, opclo, label=['open', 'close'])

    xidx = get_date_idx(len(dates), bins)

    ax.set_xticks(xidx, dates[xidx], rotation=45, ha='right')

    ax.set_title(f'Daily price range and open/close prices against time for: {ticker}')
    ax.set_xlabel('Dates')
    ax.set_ylabel('Price')
    ax.legend() # to show legend for lines

    # plt.show(block=False)


def get_date_idx(len: int, bins: int) -> list:
    bins = min(bins, len)
    return [round(pos * (len - 1) / (bins - 1)) for pos in range(bins - 1)] + [len - 1]

if __name__ == '__main__':
    main()