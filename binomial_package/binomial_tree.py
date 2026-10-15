"""
Module for path-independent option pricing using the N-period binomial model
"""
import numpy as np

def risk_neutral_prob(u, d, r):
    """
    Function returns the risk-neutral probability q = (1 + r - d) / (u - d)

    Parameters:
    u = up ratio (float)
    d = down ratio (float)
    r = risk free interest rate per period

    Condition:
    We must have d < 1 + r < u, otherwise there is an arbitrage, which breaks our model assumptions
    """
    if not (d < 1 + r < u):
        raise ValueError("inputs u,d,r must satisfy d < 1 + r < u")
    return (1+r-d)/(u-d)
    
def maturity_prices(S0, u , d, N):
    """
    The maturity price of a stock is defined as S_N = S_0 * u**k * d**(N-k), where k is the number of up moves at N periods.
    Since order does not matter here, there are N+1 possible prices at the maturity date, so this function returns a 1D Numpy array of length N+1, where entry k is the price after k up moves.

    Parameters:
    S0 = value of the stock at time 0 (float)
    u = up ratio (float)
    d = down ratio (float)
    N = number of periods (int)
    """
    k = np.arange(N+1)
    return S0 * u**k * d**(N - k)