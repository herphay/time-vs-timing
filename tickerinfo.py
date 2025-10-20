from dataclasses import dataclass


@dataclass
class TickerInfo:
    ticker:      str
    Asset_Class: str
    Sub_Class:   str
    Benchmark:   str
    Remarks:     str


def get_index_list() -> list[TickerInfo]:
    return [
        ####################
        ##### Equities #####
        ####################
        TickerInfo(
            ticker='VT',
            Asset_Class='Equities',
            Sub_Class='Global Equities (All Cap)',
            Benchmark='FTSE Global All Cap Index',
            Remarks='ETF proxy, distributing'
        ),
        TickerInfo(
            ticker='ACWI',
            Asset_Class='Equities',
            Sub_Class='Global Equities (Large/Mid Cap)',
            Benchmark='MSCI ACWI',
            Remarks='ETF proxy, distributing'
        ),
        TickerInfo(
            ticker='^990100-USD-STRD',
            Asset_Class='Equities',
            Sub_Class='Developed Equities (Large/Mid Cap)',
            Benchmark='MSCI WORLD',
            Remarks='Index, Price Return'
        ),
        TickerInfo(
            ticker='VEA',
            Asset_Class='Equities',
            Sub_Class='Developed Equities (All Cap Ex-US)',
            Benchmark='FTSE Developed All Cap ex US Index',
            Remarks='ETF proxy, distributing'
        ),
        TickerInfo(
            ticker='EFA',
            Asset_Class='Equities',
            Sub_Class='Developed Equities (Large/Mid Cap Ex-US/CA)',
            Benchmark='MSCI EAFE Index',
            Remarks='ETF proxy, distributing'
        ),
        TickerInfo(
            ticker='VWO',
            Asset_Class='Equities',
            Sub_Class='Emerging Equities (All Cap Ex-US/CA)',
            Benchmark='FTSE Emerging Markets All Cap China A Inclusion Index',
            Remarks='ETF proxy, distributing'
        ),
        TickerInfo(
            ticker='^GSPC',
            Asset_Class='Equities',
            Sub_Class='US Equities (Large Cap)',
            Benchmark='S&P500 Index',
            Remarks='Index, Price Return'
        ),
        TickerInfo(
            ticker='^SP500TR',
            Asset_Class='Equities',
            Sub_Class='US Equities (Large Cap)',
            Benchmark='S&P500 Index',
            Remarks='Index, Total Return'
        ),
        TickerInfo(
            ticker='MDY',
            Asset_Class='Equities',
            Sub_Class='US Equities (Mid Cap)',
            Benchmark='S&P Mid Cap 400 Index',
            Remarks='ETF proxy, distributing'
        ),
        TickerInfo(
            ticker='^RUT',
            Asset_Class='Equities',
            Sub_Class='US Equities (Small Cap)',
            Benchmark='Russell 2000 Index',
            Remarks='Index, Price Return'
        ),
        TickerInfo(
            ticker='IWF',
            Asset_Class='Equities',
            Sub_Class='US Equities (Large/Mid Cap Growth)',
            Benchmark='Russell 1000 Growth',
            Remarks='ETF proxy, distributing'
        ),
        TickerInfo(
            ticker='IWD',
            Asset_Class='Equities',
            Sub_Class='US Equities (Large/Mid Cap Value)',
            Benchmark='Russell 1000 Value',
            Remarks='ETF proxy, distributing'
        ),
        ########################
        ##### Fixed Income #####
        ########################
        TickerInfo(
            ticker='AGGU.L',
            Asset_Class='Fixed Income',
            Sub_Class='Global Bonds (All)',
            Benchmark='Bloomberg Global Aggregate Bond Index',
            Remarks='ETF proxy, accumulating'
        ),
        TickerInfo(
            ticker='AGG',
            Asset_Class='Fixed Income',
            Sub_Class='US Bonds (All)',
            Benchmark='Bloomberg U.S. Aggregate Bond Index',
            Remarks='ETF proxy, distributing'
        ),
        TickerInfo(
            ticker='SHY',
            Asset_Class='Fixed Income',
            Sub_Class='US Bonds (Treasuries 1-3yr)',
            Benchmark='ICE US Treasury 1-3 Year Bond Index',
            Remarks='ETF proxy, distributing'
        ),
        TickerInfo(
            ticker='IEF',
            Asset_Class='Fixed Income',
            Sub_Class='US Bonds (Treasuries 7-10yr)',
            Benchmark='ICE US Treasury 7-10 Year Bond Index',
            Remarks='ETF proxy, distributing'
        ),
        TickerInfo(
            ticker='TLT',
            Asset_Class='Fixed Income',
            Sub_Class='US Bonds (Treasuries 20+yr)',
            Benchmark='ICE US Treasury 20+ Year Bond Index',
            Remarks='ETF proxy, distributing'
        ),
        TickerInfo(
            ticker='LQD',
            Asset_Class='Fixed Income',
            Sub_Class='US Bonds (Corporate Investment Grade)',
            Benchmark='Markit iBoxx USD Liquid Investment Grade Index',
            Remarks='ETF proxy, distributing'
        ),
        TickerInfo(
            ticker='HYG',
            Asset_Class='Fixed Income',
            Sub_Class='US Bonds (Corporate High Yield)',
            Benchmark='Markit iBoxx USD Liquid High Yield Index',
            Remarks='ETF proxy, distributing'
        ),
        #######################
        ##### Real Estate #####
        #######################
        TickerInfo(
            ticker='IYR',
            Asset_Class='Real Estate',
            Sub_Class='US REIT',
            Benchmark='Dow Jones U.S. Real Estate Capped Index (USD)',
            Remarks='ETF proxy, distributing'
        ),
        TickerInfo(
            ticker='VNQI',
            Asset_Class='Real Estate',
            Sub_Class='Developed REIT (Ex-US)',
            Benchmark='S&P Global ex U.S. Property Index',
            Remarks='ETF proxy, distributing'
        ),
        TickerInfo(
            ticker='RWO',
            Asset_Class='Real Estate',
            Sub_Class='Global REIT (Developed + Emerging)',
            Benchmark='Dow Jones Global Select Real Estate Securities IndexSM',
            Remarks='ETF proxy, distributing'
        ),
        #######################
        ##### Commodities #####
        #######################
        TickerInfo(
            ticker='^SPGSCI',
            Asset_Class='Commodities',
            Sub_Class='Broad Basket Commodities',
            Benchmark='S&P GSCI',
            Remarks='Index, Total Return'
        ),
        TickerInfo(
            ticker='DJP',
            Asset_Class='Commodities',
            Sub_Class='Broad Basket Commodities',
            Benchmark='Bloomberg Commodity Index Total ReturnSM',
            Remarks='ETN, accumulating'
        ),
        TickerInfo(
            ticker='IAU',
            Asset_Class='Commodities',
            Sub_Class='Gold',
            Benchmark='LBMA Gold Price',
            Remarks='ETF, accumulating'
        ),
        TickerInfo(
            ticker='GC=F',
            Asset_Class='Commodities',
            Sub_Class='Gold',
            Benchmark='Gold Futures',
            Remarks='Gold Futures Contract'
        ),
    ]

def get_mag7() -> list[TickerInfo]:
    return [
        TickerInfo(
            ticker='GOOG',
            Asset_Class='Equities',
            Sub_Class='US Equities (Large Cap)',
            Benchmark='S&P500 Index',
            Remarks='MAG7, SP500'
        ),
        TickerInfo(
            ticker='AMZN',
            Asset_Class='Equities',
            Sub_Class='US Equities (Large Cap)',
            Benchmark='S&P500 Index',
            Remarks='MAG7, SP500'
        ),
        TickerInfo(
            ticker='AAPL',
            Asset_Class='Equities',
            Sub_Class='US Equities (Large Cap)',
            Benchmark='S&P500 Index',
            Remarks='MAG7, SP500'
        ),
        TickerInfo(
            ticker='META',
            Asset_Class='Equities',
            Sub_Class='US Equities (Large Cap)',
            Benchmark='S&P500 Index',
            Remarks='MAG7, SP500'
        ),
        TickerInfo(
            ticker='MSFT',
            Asset_Class='Equities',
            Sub_Class='US Equities (Large Cap)',
            Benchmark='S&P500 Index',
            Remarks='MAG7, SP500'
        ),
        TickerInfo(
            ticker='NVDA',
            Asset_Class='Equities',
            Sub_Class='US Equities (Large Cap)',
            Benchmark='S&P500 Index',
            Remarks='MAG7, SP500'
        ),
        TickerInfo(
            ticker='TSLA',
            Asset_Class='Equities',
            Sub_Class='US Equities (Large Cap)',
            Benchmark='S&P500 Index',
            Remarks='MAG7, SP500'
        ),
    ]