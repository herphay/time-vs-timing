import matplotlib.pyplot as plt
import numpy as np
from data import pull_ticker_data

def main() -> None:
    ...


def plot_high_low(ticker: str, start: str = None, end: str = None) -> None:
    """
    ticker: str
    start/end: str
        ISO 8601 (YYYY-MM-DD)
    """
    data = pull_ticker_data(ticker, cols='date, high, low', start=start, end=end)
    dates, highs, lows = zip(*data)
    dates = np.array(dates)
    highs = np.array(highs)
    lows = np.array(lows)

    fig, ax = plt.subplots()
    ax.fill_between(dates, highs, lows, alpha=.5, linewidth=0)
    ax.plot(dates, (highs + lows)/2, linewidth=2)

    plt.show()


if __name__ == '__main__':
    main()