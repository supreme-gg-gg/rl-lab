import numpy as np
import cvxpy as cp
import yfinance as yf

symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
data = yf.download(symbols, start='2020-01-01', end='2023-01-01')['Adj Close']

rets = np.log(data/data.shift(1))
mu = rets.mean().values
cov = rets.cov().values
n = len(data.columns)

# Risk adjusted return takes into account profit as well

risk_aversion = 0.5

weights = cp.Variable(n)
risk_adjusted = mu @ weights - risk_aversion * cp.quad_form(weights, cov)

objective = cp.Maximize(risk_adjusted)
constraints = [
    cp.sum(weights) == 1,
    weights >= 0
]

prob = cp.Problem(objective, constraints)
prob.solve()

portfolio_return = mu @ weights
portfolio_risk = cp.quad_form(weights, cov)

print(f"Optimal portfolio weights: {weights.value}")
print(f"Expected portfolio return: {portfolio_return.value}")
print(f"Portfolio risk (variance): {portfolio_risk.value}")

import matplotlib.pyplot as plt

plt.bar(data.columns, weights.value)
plt.title('Optimal Portfolio Weights')
plt.show()
