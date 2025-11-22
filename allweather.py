import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from typing import Literal

from helpers import ticker_data2df
from montecarlo import get_ticker_stats

# https://www.ssga.com/us/en/intermediary/etfs/spdr-bridgewater-all-weather-etf-allw
# https://www.quantifiedstrategies.com/ray-dalios-all-weather-portfolio/
# https://www.lazyportfolioetf.com/allocation/ray-dalio-all-weather/

# Allocations:
# Retail no leverage: 


def all_weather_retail(
        data_start: str | None = None,
        data_end: str | None = None,
        data_freq: Literal['D', 'M'] = 'D',
        period: str = '5y',
        components: list[str] = ['VTI', 'TLT', 'IEI', 'DBC', 'GLD'],
        weights: np.ndarray = np.array([0.3, 0.4, 0.15, 0.075, 0.075]),
        return_type: Literal['log', 'simple'] = 'log',
        bounds: np.ndarray = np.array([1.5, 2.5]),
        plot_return_type: Literal['log', 'simple'] = 'simple',
        plot: bool = True
    ) -> None:
    """
    All Weather Retail portfolio
    Suggested portfolio weights
        VTI - 30%
        TLT - 40%
        IEI - 15%
        DBC - 7.5%
        GLD - 7.5%
    """
    # Preset portfolio metrics
    mu, cov, prices = get_ticker_stats(components, price_freq=data_freq, return_type=return_type,
                                       start=data_start, end=data_end)

    # portfolio Expected Returns
    pmu = mu @ weights
    # portfolio volatility
    vol = np.sqrt(weights.T @ cov @ weights) # weights is 1d, no need to transpose to w.T @ cov @ w

    print(f'Portfolio Expected Return is {pmu} per {data_freq}, vol is {vol}')

    unit = period[-1]
    period = int(period[:-1])

    conversion_dict = {'y': {'M': 12, 'D': 252}, 'm': {'M': 1, 'D': 21}}
    sim_t = np.arange(period * conversion_dict[unit][data_freq] + 1)
    E_returns = pmu * sim_t
    print(E_returns)
    
    bounds.sort()
    bounds = np.append(bounds[::-1], -bounds)
    E_bounds = bounds[:, None] @ np.sqrt(sim_t)[None, :] * vol + E_returns
    
    expectations = np.vstack([E_bounds, E_returns])


    if plot:
        # print(np.log1p(prices.pct_change()).to_numpy()[-len(sim_t):].shape)
        realized = np.log1p(prices.pct_change()).to_numpy()[-len(sim_t):].cumsum(axis=0)
        expectations = np.vstack([expectations, realized.T])
        if return_type == 'log' and plot_return_type == 'simple':
            expectations = np.e ** expectations

        plt.plot(expectations.T)
        plt.title(f'Returns over')
        plt.show()
    
    return expectations, realized.T, prices




if __name__ == '__main__':
    ...