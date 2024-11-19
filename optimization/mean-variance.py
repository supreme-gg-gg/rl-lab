import cvxpy as cp
import numpy as np
import pandas as pd
import yfinance as yf

# MVO only cares about portfolio risk, as long as it reaches a min target return...

# https://www.kaggle.com/code/marketneutral/cvxpy-portfolio-optimization-example/notebook

symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
data = yf.download(symbols, start='2021-01-01', end='2023-01-01')['Adj Close']

rets = np.log(data / data.shift(1))
mu = rets.mean().values
cov = rets.cov().values
n = len(data.columns)

average_return = np.mean(mu)
target_return = 0.9 * average_return
risk_aversion = 0.5

weights = cp.Variable(n)
portfolio_return = mu @ weights
portfolio_risk = cp.quad_form(weights, cov)

objective = cp.Minimize(portfolio_risk)
constraints = [
    cp.sum(weights) == 1,
    portfolio_return >= target_return,
    weights >= 0
]

prob = cp.Problem(objective, constraints)
prob.solve()

print(f"Optimal portfolio weights: {weights.value}")
print(f"Expected portfolio return: {portfolio_return.value}")
print(f"Portfolio risk (variance): {portfolio_risk.value}")

import matplotlib.pyplot as plt

plt.bar(data.columns, weights.value)
plt.title('Optimal Portfolio Weights')
plt.show()