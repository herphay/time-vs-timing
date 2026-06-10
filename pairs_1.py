import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

from helpers import ticker_data2df

def catl():
    catl = ticker_data2df(['300750.SZ', '3750.HK', 'CNYHKD=X'])
    catl.columns = ['SZ_A', 'HK_H', 'CNYHKD']

    catl['SZ_A'] = catl['SZ_A'] * catl['CNYHKD']

    catl['premium'] = catl['HK_H'] / catl['SZ_A'] - 1

    plt.figure()
    ax1 = catl['premium'].plot()
    
    ax1.minorticks_on()
    ax1.grid(which='major', color='dimgray', alpha=0.75)
    ax1.grid(which='minor', color='gray', alpha=0.2)
    date_form = mdates.DateFormatter('%Y-%m-%d')
    ax1.xaxis.set_major_formatter(date_form)
    ax1.set_title('CATL H to A premium (nominal price)')

    spread = catl['premium']

    calculate_cointegration(spread)
    print('\n', '#' * 45, '\n')
    half_life = calculate_halflife(spread)
    catl['z_score'], catl['rolling_vol'] = calculate_zscore(spread, window=round(half_life))
    max_drawdown, min_premium = calculate_maximum_adverse_excursion(spread)
    print('\n', '#' * 45, '\n')

    print('Previous 15 days A/H premium')
    print(catl[-15:])

    # Find the payout ratio:
    # PnL = V0 * (PA1 / PA0) * [(R0 - R1) / (1 + R0)]
    # V0 is single leg size
    # PA0, PA1 is price of A share on entry and exit accordingly
    # R0, R1 is the price premium on entry and exit accordingly
    current_premium = catl['premium'].iloc[-1]
    exit_premiums = np.arange(round(min_premium, 2), 
                              round(current_premium + max_drawdown + 0.01, 2), 
                              step=0.01)
    # Assumes no price appreciation/depreciation of the underyling
    beta_neutral_payout = (current_premium - exit_premiums) / (1 + current_premium)
    plt.figure()
    plt.plot(exit_premiums, beta_neutral_payout, label='payout')
    plt.plot([current_premium] * len(exit_premiums), beta_neutral_payout, 
             label='Current premium', linestyle='--', alpha=0.5)
    plt.plot([min_premium] * len(exit_premiums), beta_neutral_payout,
             label=f'Historical min ({(current_premium - min_premium) / (1 + current_premium):.2%} gain)', 
             linestyle='--', alpha=0.5, color='limegreen')
    max_premium = catl['premium'].max()
    plt.plot([max_premium] * len(exit_premiums), beta_neutral_payout,
             label=f'Historical max ({(max_premium - current_premium) / (1 + current_premium):.2%} loss)', 
             linestyle='--', alpha=0.5, color='red')
    plt.title(f'PnL curve - {0.01 / (1 + current_premium):.3%} delta per % spread move')
    plt.minorticks_on()
    plt.grid(which='major', color='dimgray', alpha=0.75)
    plt.grid(which='minor', color='gray', alpha=0.2)
    plt.legend()

    return catl


def legend_lenovo(
        start: str = '2023-04-01'
    ):
    """Legend Holding vs Lenovo pairs trade"""
    lenovo = ticker_data2df(['3396.HK', '0992.HK'], start=start)
    lenovo.columns = ['Legend', 'Lenovo']

    # shares_outstanding = np.array([2_356_230_000, 12_404_659_302])
    # lenovo[['Legend_NAV', 'Lenovo_NAV']] = lenovo[['Legend', 'Lenovo']] * shares_outstanding
    # lenovo['holding_NAV'] = lenovo['Lenovo_NAV'] * 0.3141

    NAVratio = 0.3141 * 12_404_659_302 / 2_356_230_000

    lenovo['discount_ratio'] = 1 - lenovo['Legend'] / (lenovo['Lenovo'] * NAVratio)
    
    lenovo['discount_amt'] = (lenovo['Lenovo'] * NAVratio) - lenovo['Legend']
    
    fig, ax = plt.subplots(3,1,figsize=(6.4, 10))

    (lenovo[['Legend', 'Lenovo']] / lenovo[['Legend', 'Lenovo']].iloc[0]).plot(ax=ax[0], title='Legend vs Lenovo')
    lenovo['discount_ratio'].plot(ax=ax[1], title='Legend discount ratio')
    lenovo['discount_amt'].plot(ax=ax[2], title='Legend discount amount')

    for a in ax:
        a.minorticks_on()
        a.grid(which='major', color='dimgray', alpha=0.75)
        a.grid(which='minor', color='gray', alpha=0.2)
    plt.tight_layout()

    calculate_cointegration(lenovo['discount_ratio'])
    print('\n', '#' * 45, '\n')
    half_life = calculate_halflife(lenovo['discount_ratio'])
    lenovo['z_score'], lenovo['rolling_vol'] = calculate_zscore(lenovo['discount_ratio'], 
                                                                window=round(half_life))
    max_drawdown, min_premium = calculate_maximum_adverse_excursion(lenovo['discount_ratio'])
    print('\n', '#' * 45, '\n')

    print('PnL Formula')
    print('PnL = Single leg Capital * dS * P_hold_0 / NAVratio / P_sub_0 * hold_return')
    current_spread = lenovo['discount_ratio'].iloc[-1]
    max_spread = lenovo['discount_ratio'].max()
    stop_spread = max_spread - current_spread
    print(f'Current Spread: {current_spread:.2%} vs Peak Spread: {max_spread:.2%}')
    print(f'The NAV Ratio is: {NAVratio:.2f}\n')

    unit_PnL = 0.01 * lenovo['Legend'].iloc[-1] / lenovo['Lenovo'].iloc[-1] / NAVratio * 1000

    print(f'Unit PnL per % per k single leg capital is: ${unit_PnL:.2f} HKD')
    print('### Unit PnL scales by Holding company returns too (end price/start price ratio) ###')

    print(f'Potential loss if expand to max ({stop_spread:.2%}) is ' + 
          f'${100 * stop_spread * unit_PnL:.2f} HKD per k')

    return lenovo


def kingboard(
        start: str = None
    ):
    """Kingboard Holdings vs Kingboard Laminates pair trade"""
    lenovo = ticker_data2df(['0148.HK', '1888.HK'], start=start)
    lenovo.columns = ['Holdings', 'Laminates']

    NAVratio = 0.711 * 3_135_325_000 / 1_108_311_736

    lenovo['discount_ratio'] = 1 - lenovo['Holdings'] / (lenovo['Laminates'] * NAVratio)
    
    lenovo['discount_amt'] = (lenovo['Laminates'] * NAVratio) - lenovo['Holdings']
    
    fig, ax = plt.subplots(3,1,figsize=(6.4, 10))

    (lenovo[['Holdings', 'Laminates']] / 
     lenovo[['Holdings', 'Laminates']].iloc[0]).plot(ax=ax[0], title='Holdings vs Laminate')
    lenovo['discount_ratio'].plot(ax=ax[1], title='Kingboard discount ratio')
    lenovo['discount_amt'].plot(ax=ax[2], title='Kingboard discount amount')

    for a in ax:
        a.minorticks_on()
        a.grid(which='major', color='dimgray', alpha=0.75)
        a.grid(which='minor', color='gray', alpha=0.2)
        print(a)
    plt.tight_layout()

    calculate_cointegration(lenovo['discount_ratio'])
    print('\n', '#' * 45, '\n')
    half_life = calculate_halflife(lenovo['discount_ratio'])
    lenovo['z_score'], lenovo['rolling_vol'] = calculate_zscore(lenovo['discount_ratio'], 
                                                                window=round(half_life))
    max_drawdown, min_premium = calculate_maximum_adverse_excursion(lenovo['discount_ratio'])
    print('\n', '#' * 45, '\n')

    print('PnL Formula')
    print('PnL = Single leg Capital * dS * P_hold_0 / NAVratio / P_sub_0 * hold_return')
    current_spread = lenovo['discount_ratio'].iloc[-1]
    max_spread = lenovo['discount_ratio'].max()
    stop_spread = max_spread - current_spread
    print(f'Current Spread: {current_spread:.2%} vs Peak Spread: {max_spread:.2%}')
    print(f'The NAV Ratio is: {NAVratio:.2f}\n')

    unit_PnL = 0.01 * lenovo['Holdings'].iloc[-1] / lenovo['Laminates'].iloc[-1] / NAVratio * 1000

    print(f'Unit PnL per % per k single leg capital is: ${unit_PnL:.2f} HKD')
    print('### Unit PnL scales by Holding company returns too (end price/start price ratio) ###')

    print(f'Potential loss if expand to max ({stop_spread:.2%}) is ' + 
          f'${100 * stop_spread * unit_PnL:.2f} HKD per k')
    
    return lenovo


def calculate_cointegration(
        series: pd.Series
    ):
    # Perform Augmented Dickey-Fuller test
    adf_result = adfuller(series.dropna())
    
    adf_statistic = adf_result[0]
    p_value = adf_result[1]
    critical_values = adf_result[4]
    
    print(f"ADF Statistic: {adf_statistic:.4f}")
    print(f"P-Value: {p_value:.4f}")
    print("Critical Values:")
    for key, value in critical_values.items():
        print(f"  {key}: {value:.4f}")
        
    if p_value < 0.05:
        print("Conclusion: The series is stationary (Cointegrated).")
    else:
        print("Conclusion: The series is non-stationary (Risk of permanent divergence).")


def calculate_halflife(
        series: pd.Series
    ):
    # Calculate the lagged series and the change in the series
    series_lag = series.shift(1).dropna()
    series_diff = series.diff().dropna()
    
    # Align the data after dropping NaNs
    df_temp = pd.concat([series_lag, series_diff], axis=1)
    df_temp.columns = ['lag', 'diff']
    
    # Add a constant for the OLS regression
    X = sm.add_constant(df_temp['lag'])
    y = df_temp['diff']
    
    # Fit the OLS model
    model = sm.OLS(y, X).fit()
    
    # Extract the lambda (coefficient of the lagged spread)
    lambda_val = model.params['lag']
    
    # Calculate half-life
    half_life = -np.log(2) / lambda_val
    
    print(f"Mean Reversion Lambda: {lambda_val:.4f}")
    print(f"Half-Life: {half_life:.2f} periods (days)")
    
    return half_life


def calculate_zscore(
        series: pd.Series, 
        window: int = 20
    ):
    # Calculate rolling mean and standard deviation
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    
    # Calculate Z-score
    z_score = (series - rolling_mean) / rolling_std
    
    return z_score, rolling_std


def calculate_maximum_adverse_excursion(
        series: pd.Series,
        show_details: bool = False
    ):
    # For an arbitrageur shorting the premium, risk is when the premium goes UP.
    # We calculate the max peak-to-trough expansion in the premium.
    
    # Calculate the running minimum (best entry point for a blowout)
    series = series.iloc[29:]
    running_min = series.cummin()
    if show_details:
        print('Cumulative min premium')
        print(running_min, '\n')
    
    # Calculate the expansion from the running minimum
    drawdown = series - running_min
    if show_details:
        print('Drawdown to the cummin')
        print(drawdown, '\n')
    
    # Find the maximum expansion
    max_drawdown = drawdown.max()
    
    print(f'Running minimum currently is: {running_min.iloc[-1] * 100:.2f}% occuring on ' + 
          f'{series.idxmin().strftime('%Y-%m-%d')}')
    print(f'Max of {series[drawdown.idxmax()] * 100:.2f}% happens on ' +
          f'{drawdown.idxmax().strftime('%Y-%m-%d')}')
    print(f"Maximum Spread Expansion (Drawdown Risk): +{max_drawdown * 100:.2f}%")
    
    return max_drawdown, running_min.iloc[-1]