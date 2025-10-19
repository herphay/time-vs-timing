import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

    return mean_return.to_numpy(), covariance_matrix.to_numpy(), data


def montecarlo_sim(
        meanReturn: pd.Series,
        covarianceMatrix: pd.DataFrame,
        stockWeights: np.ndarray,
        sims: int = 100,
        timeframe: int = 100
    ) -> None:
    """
    Monte Carlo simulation of stock prices -> QuantPy implementation

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
    num_components = len(meanReturn) # number of component securities
    # build a meanReturn matrix where meanReturn for each security is copied to all simulated days
    meanReturns = np.full(shape=(timeframe, num_components), fill_value=meanReturn)
    meanReturns = meanReturns.T # stock * days
    # build a final returns matrix where rows are each day, and columns are each simulation run
    # This matrix will store the final simulated cumulative returns
    portfolio_sims = np.full(shape=(timeframe, sims), fill_value=0.0) # days * sim#

    rng = np.random.default_rng() # set RNG seed once rather than for each sim round

    for i in range(sims):
        # Get the lower triangle matrix
        L = np.linalg.cholesky(covarianceMatrix)         # stock * stock
        # Get independent normal random samples for each stock across all days
        Z = rng.normal(size=(timeframe, num_components)) # days  * stock
        # calc daily returns for each stock across all days -> make returns correlated per covariance
        dailyreturns = meanReturns + np.inner(L, Z)      # stock * days
        # stockWeights is 1 * stock, thus dailyreturns need to be changed to days * stock for np.inner
        # Result will be 1 * days of total portfolio returns 
        ############# Biggest assumption is that portfolio weights are rebalanced each day to match
        ############# initial weights
        portfolio_sims[:, i] = np.cumprod(1 + np.inner(stockWeights, dailyreturns.T))

    print(portfolio_sims.shape)
    plt.plot(portfolio_sims)
    plt.show()