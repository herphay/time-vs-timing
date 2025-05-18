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
        cashflows
    ) -> float:
    ...


def xnpv(
        cashflows
    ) -> float:
    ...


def xnpv_prime(
        cashflows
    ) -> float:
    ...