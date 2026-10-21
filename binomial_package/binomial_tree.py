###
## Binomial Model for Option Pricing
###


## This is a module for path-independent option pricing using the N-period binomial model

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


def call_payoff(S, K):
    """
    The payoff function for a European call is max(S_N - K, 0).
    
    Parameters:
    S = stock prices at maturity (numpy.ndarray)
    K = strike price (float)
    """
    return np.maximum(S - K, 0)


def put_payoff(S, K):
    """
    The payoff function for a European put is max(K - S_N, 0)

    Parameters:
    S = stock prices at maturity (numpy.ndarray)
    K = strike price (float)
    """
    return np.maximum(K - S, 0)


def binary_call_payoff(S, K):
    """
    The payoff function for a binary call is I(S_N >= K) where I is the indicator function

    Parameters:
    S = stock prices at maturity (numpy.ndarray)
    K = strike price (float)
    """
    return np.where(S >= K, 1.0, 0.0)


def binary_put_payoff(S, K):
    """
    The payoff function for a binary put is I(S_N <= K) where I is the indicator function

    Parameters:
    S = stock prices at maturity (numpy.ndarray)
    K = strike price (float)
    """
    return np.where(S <= K, 1.0, 0.0)


def option_price(S0, K, u, d, r, N, payoff):
    """
    This function prices a path-dependent option using the N-period binomial model.
    
    This function uses backward induction to find C_0:

    Start from the payoffs at maturity, C_N = f(S_N), then step back one period at a time. 
    On the tree, k up-moves at time n moves to k+1 up-moves at time n+1 after an up-move (H) and stays on k up-moves after a down-move (T), so

    C_n[k] = (q * C_{n+1}[k+1] + (1 - q) * C_{n+1}[k]) / (1 + r)

    Parameters:
    S0 = initial stock price (float)
    K = strike price (float)
    u = up factor per period (float)
    d = down factor per period (float)
    r = risk-free interest rate per period (float)
    N = number of periods (int)
    payoff = function f(S, K) giving payoffs at maturity, choose 1 input from call_payoff, put_payoff, binary_call_payoff, binary_put_payoff
    """
    q = risk_neutral_prob(u, d, r)
    V = payoff(maturity_prices(S0, u, d, N), K)
    for _ in range(N):
        up = V[1:]  # copy of V without its first entry
        down = V[:-1] # copy of V without its last entry
        V = (q * up + (1 - q) * down) / (1 + r)
    return V[0]


def estim_parameters(prices):
    """
    Since real data doesnt go up or down by deterministic factors, we must estimate the parameters in our binomial model.

    To do this simply, we return:
    u: the average factor for 'up days'
    d: the average factor for 'down days'
    p: the proportion of up days 

    The input 'prices' is the dataset of closing prices (must be oldest to newest).
    """
    ratios = prices[1:] / prices [:-1] # sliced array including everything but the first entry, divded by a sliced array including everything but the last entry

    up_days = ratios > 1
    down_days = ~up_days

    u = np.mean(ratios[up_days])
    d = np.mean(ratios[down_days])

    p = np.mean(up_days)

    return u, d, p