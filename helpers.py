import numpy as np
import pandas as pd
from scipy.optimize import newton
from collections.abc import Iterable

from data import get_all_tickers

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

    # Get the powers for the derivative of the XNPV function
    deri_powers = powers - 1

    # Get the coefficients for the derivative of the XNPV function
    deri_coeff = cashflows * powers

    # call scipy.optimize.newton and pass the XNPV func along with its 1st derivative
    # Here, solve for x where x = 1 / (1 + xirr)
    inv_r = newton(
                func=lambda x: sum([c * x ** p for c, p in zip (cashflows, powers)]),
                x0 = 1 / (1 + rate_guess),
                fprime=lambda x: sum([c * x ** p for c, p in zip(deri_coeff, deri_powers)])
            )

    return 1 / inv_r - 1


#%%
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