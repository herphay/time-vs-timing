import matplotlib.pyplot as plt
import numpy as np
from data import pull_ticker_data

def main() -> None:
    plot_day_range('VT', start='2025-01-01')


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
    bins = min(bins, len(dates))
    # Manual calc (round to int)
    xidx = np.array([round(pos * (len(dates) - 1)/(bins - 1)) for pos in range(bins - 1)] + [len(dates) - 1])
    
    # Using Numpy linspace (round down to int)
    # x_idx2 = np.array(np.linspace(0, len(dates)-1, 21, dtype=int))
    
    ax.set_xticks(xidx, dates[xidx], rotation=45, ha='right')
    # ax.tick_params('x', rotation=90)   # redundant, another way to set rotation
    # fig.autofmt_xdate()                # Auto rotate dates but does not properly increase tick gap

    plt.show()


if __name__ == '__main__':
    main()