import pandas as pd
import numpy as np

from collections.abc import Iterable
from datetime import datetime, timedelta

from helpers import ticker_data2df

def get_ticker_stats(
        tickers: Iterable[str],
        start: str | None = None,
        end: str | None = None,
    ) -> tuple[pd.Series, pd.DataFrame]:
    data = ticker_data2df(tickers=tickers, start=start, end=end)

    returns = data.pct_change()
    mean_return = returns.mean()
    covariance_matrix = returns.cov()

    return mean_return, covariance_matrix


def montecarlo_sim(
        meanReturn: pd.Series,
        covarianceMatrix: pd.DataFrame,
        stockWeights: np.ndarray,
        sims: int,
        timeframe: int
    ) -> None:
    """
    Monte Carlo simulation of stock prices

    meanReturn: pd.Series
        average return of the stocks involved
    covarianceMatrix: pd.DataFrame
        covariance matrix of the returns of the stocks
    stockWeights: np.ndarray
        weight of each stock in the portfolio
    sims: int
        number of simulations
    timeframe:
        number of days to simulate
    """

