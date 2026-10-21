###
## Module for monte carlo simulations
###

## FTAP: C = (1+r)^-N * E[C_N]
## Its not as simple to calculate the expected payoff of a path-dependent option, such as an Asian call option, so we use a Monte Carlo approximation:
## 1. Simulate many price paths, with up-moves happening with probability q
## 2. Compute each paths payoff
## 3. Average the payoffs and discount by (1+r)^-N

import numpy as np
from binomial_package.binomial_tree import risk_neutral_prob


def simulate_paths(S0, u, d, r, N, n_paths, seed=None):
    """
    Simulating stock price paths in the binomial model under the risk-neutral probability
    
    Returns an array of shape (n_paths, N+1) where the row i is path i and column j is the value of the stock price at path step j
    """
    q = risk_neutral_prob(u, d, r)
    rng = np.random.default_rng(seed)

    ups = rng.random((n_paths, N)) < q  # turns the table of unif(0,1) into true/ false coin tosses with P(up) = q

    factors = np.where(ups, u, d) # turns true into u and falses in to d, so we know if its an up move or a down move along each path step

    paths = S0 * np.cumprod(factors, axis = 1) # gives us the cumulative price of the stock for each path

    initial = np.full((n_paths,1), S0)

    return np.concatenate((initial, paths), axis = 1)


def asian_call_payoff(paths, K):
    """
    The payoff of an Asian call option is max(A_N - K, 0) where A_N is the average value of the stock price along the path.
    
    Returns an array of shape (n_paths,) where entry i is the payoff of path i
    """
    average = np.mean(paths, axis = 1)
    
    return np.maximum(average - K, 0)


def mc_price(payoffs, r , N):
    """
    Monte carlo estimate of the no-arbitrage price according to the FTAP, using the average of simulated payoffs.

    Returns the estimated price.
    """
    discount = (1 + r)**(-N)

    price = discount * np.mean(payoffs)

    return price


    



