"""
Module for option pricing using the N-period binomial model
"""

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
    
