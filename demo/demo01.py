### Demo 01

"""
This demo checks the package against worked examples from the MA52100 notes, found in data/MA52100_examples.csv. 
Each example is priced with the method in its 'method' column, and compared with the answer from the notes.
Path-independent methods use the binomial tree module, whereas path-dependent methods use the monte carlo module.

This demo can be used from the projects root folder with:
    python -m demo.demo01
"""

from binomial_package import option_price, call_payoff, put_payoff, binary_call_payoff, simulate_paths, asian_call_payoff, mc_price

payoffs = {
    "call": call_payoff,
    "put": put_payoff,
    "binary_call": binary_call_payoff,
    "asian_call": asian_call_payoff,
    }

n_paths = 100000

print(f"{'Example':<8} {'Option':<12} {'Method':<6} {'Lec Notes':>10} {'Package':>12}")

with open("data/MA52100_examples.csv", "r") as f:
    f.readline()
    for line in f:
        example, option, method, S0, K, u, d, r, N, answer = line.strip().split(",")

        S0, K = float(S0), float(K)
        u, d, r = float(u), float(d), float(r)
        N = int(N)
        answer = float(answer)
        payoff = payoffs[option]
                
        if method == "tree":
            price = option_price(S0, K, u, d, r, N, payoff)
        else:
            paths = simulate_paths(S0, u, d, r, N, n_paths, seed=1)
            price = mc_price(payoff(paths, K), r, N)

        print(f"{example:<8} {option:<12} {method:<6} {answer:>10.4f} {price:>12.4f}")




