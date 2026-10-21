### Demo 02

"""
If the FTSE 100 moves like it did over the past year, whats the no-arbitrage price for a European call option expiring in 3-months time?

The demo estimates u, d, and p from a year of daily FTSE 100 prices, from data/FTSE100.csv, and then prices a 3-month 'at the money' European call option.
It then plots the Monte Carlo estimate converging to that price as the number of simulated paths increases.

'At-the-money' means the strike price is equal to the stock price today

Run the file from the projects root folder with:
    python -m demo.demo01
"""
import numpy as np
import matplotlib.pyplot as plt

from binomial_package import estim_parameters, risk_neutral_prob, option_price, call_payoff, simulate_paths, mc_price

prices = np.genfromtxt("data/FTSE100.csv", delimiter=",", skip_header=1, usecols=1)

u, d, p = estim_parameters(prices)

# Assumed an interest rate of 4% a year spread over 252 trading days, so the daily risk-free rate is:
r = 0.04 / 252

q = risk_neutral_prob(u, d, r)

print(f"Estimated from {len(prices)} daily prices:")
print(f"u = {u:.4f}, d = {d:.4f}, p = {p:.4f}")
print(f"Daily risk-free rate r = {r:.4f}, risk-neutral q = {q:.4f}")

S0 = prices[-1]
K = S0
N = 63 # 3 months is only 252/4 = 63 trading days

model_price = option_price(S0, K, u, d, r, N, call_payoff)
print(f"\nAt-the-money call, S0 = K = {S0:.2f}, N = {N} days")
print(f"Binomial tree price: {model_price:.4f}")

n_paths_list = [100, 300, 1000, 3000, 10000, 30000, 100000]
estimates = []

for n_paths in n_paths_list:
    paths = simulate_paths(S0, u, d, r, N, n_paths, seed=1)
    payoffs = call_payoff(paths[:, -1], K)
    estimates.append(mc_price(payoffs, r, N))
    print(f"{n_paths} paths: {estimates[-1]:.4f}")


plt.plot(n_paths_list, estimates, "o-", label="Monte Carlo estimate")
plt.axhline(model_price, color="black", linestyle="--", label="Binomial model price")
plt.xscale("log")
plt.xlabel("Number of simulated paths")
plt.ylabel("Call price")
plt.title("Monte Carlo price converging to the binomial model price")
plt.legend()
plt.show()



