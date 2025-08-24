from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

def test_cointegration(y, x):
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    spread = model.resid
    pvalue = adfuller(spread)[1]
    return pvalue, model.params[1], spread
