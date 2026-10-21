# Binomial Option Pricing

This is a Python package for pricing options in the discrete-time N-period binomial model.

The package prices options with 2 methods:

- **Binomial tree model**: this finds the no-arbitrage prices for different European options,
whose payoff depends only on the final stock price (path-independent).
- **Monte Carlo simulation**: this estimates the price, according to the FTAP, by simulating stock price paths under the risk-neutral probability q,
and averaging the discounted payoffs. This can also handle path-dependent options such as the Asian call, which cannot be priced on a tree so easily.

The package can also estimate the model parameters u, d, and p from historical price data, so the model can be applied to a real index.

Note: u, d, and r are all per period.

## Installing the package

This package uses only the Python standard library, NumPy, and Matplotlib. The steps to install the package are as follows:

1. Clone the repository and move into it:

```bash
git clone https://github.com/NoahParker04/binomial_option_pricing.git
cd binomial_option_pricing
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows
```

3. Install the package in editable mode:

```bash
pip install -e .
```
The required dependencies are specified in `pyproject.toml` and will be installed automatically.

## Using the package

All main functions can be imported directly from the package.

Here are some examples of usage:

1. Using the binomial model

```python
from binomial_package import option_price, call_payoff

# European call with S_0 = 2025, K = 1800, u = 4/3, d = 5/6, r = 0.25, N = 2

price = option_price(2025, 1800, 4/3, 5/6, 0.25, 2, call_payoff)

print(price)    # 880.0
```

2. Monte Carlo pricing

```python
from binomial_package import simulate_paths, asian_call_payoff, mc_price

# Asian call with S_0 = 1, K = 1, u = 2, d = 0.5, r = 0.25, N = 3

paths = simulate_paths(1, 2, 0.5, 0.25, 3, 100000, seed=1)

price = mc_price(asian_call_payoff(paths, 1), 0.25, 3)

print(price)    # approximately 0.304
```
## Demo scripts

- **demo01**: prices worked examples from the MA52100 notes and prints the lecture notes answers next to the package's results.
- **demo02**: estimates u, d and p from FTSE 100 data, prices a 3-month at-the-money call on the tree, and plots the Monte Carlo price converging to it.

Run the demos from the projects root folder with:

```bash
python -m demo.demo01
python -m demo.demo02
```

## Data

- **data/MA52100_examples.csv**: these are parameters and answers for worked examples from the MA52100 lecture notes, used by demo01.
- **data/FTSE100.csv**: daily FTSE 100 closing prices from 14/10/2025 to 14/10/2026, downloaded from https://uk.investing.com/indices/uk-100-historical-data.








