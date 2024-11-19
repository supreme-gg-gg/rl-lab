from agent import DQNAgent
import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch

class AgentPortfolio(DQNAgent):

    def __init__(self, env, device, model_to_load=None, balance=100_000):
        super().__init__(env, device, model_to_load=model_to_load)
        self.initial_portfolio_value = balance
        self.balance = balance
        self.inventory = []
        self.return_rates = []
        self.portfolio_values = [balance]
        self.buy_dates = []
        self.sell_dates = []

        test_logger = logging.getLogger('test_model')
        test_logger.setLevel(logging.INFO)
        test_handler = logging.FileHandler('logs/test.log', mode='w')
        test_formatter = logging.Formatter('[%(asctime)s] %(levelname)s: [TEST] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        test_handler.setFormatter(test_formatter)
        test_logger.addHandler(test_handler)
        self.logger = test_logger

    def reset_portfolio(self):
        self.balances = self.initial_portfolio_value
        self.inventory = []
        self.return_rates = []
        self.portfolio_values = [self.initial_portfolio_value]
    
    def hold(self):
        self.logger.info("Hold")
    
    def buy(self, price, t):
        self.balance -= price
        self.inventory.append(price)
        self.buy_dates.append(t)
        self.logger.info('Buy:  ${:.2f}'.format(price))

    def sell(self, price, t):
        self.balance += price
        bought_price = self.inventory.pop(0)
        profit = price - bought_price
        global reward
        reward = profit
        self.sell_dates.append(t)
        self.logger.info('Sell: ${:.2f} | Profit: ${:.2f}'.format(price, profit))

    def test_agent(self, num_episodes):

        '''
        Main entry point for testing the DQN Agent!
        '''

        self.train_period = num_episodes

        for i in range(num_episodes):
            self.episode = 7
            state = self.env.reset()[0]
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            t = 0
            done = False
            
            while not done:
                
                # In test mode turn off epsilon greedy policy
                action = self.select_action(state, train=False)
                action = action.item()
                next_state, _, done, _,  _ = self.env.step(action)
                unscaled_state = next_state[-1] * self.env.std + self.env.mean # undo z-score
                stock_price = unscaled_state[3] # Remeber that state buffer is 2D!!
                next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                prev_portfolio_value = len(self.inventory) * stock_price + self.balance

                if action == 0: self.hold() # hold
                if action == 1 and self.balance > stock_price: self.buy(stock_price, t) # buy
                if action == 2 and len(self.inventory) > 0: self.sell(stock_price, t) # sell

                current_portfolio_value = len(self.inventory) * stock_price + self.balance

                self.return_rates.append((current_portfolio_value - prev_portfolio_value) / prev_portfolio_value)
                self.portfolio_values.append(current_portfolio_value)

                state = next_state

                t += 1
            
            _ = self.evaluate_portfolio_performance()
            self.plot_portfolio()
    
    def evaluate_portfolio_performance(self):
        
        portfolio_return = self.portfolio_values[-1] - self.initial_portfolio_value
        
        self.logger.info("--------------------------------")
        self.logger.info('Portfolio Value:        ${:.2f}'.format(self.portfolio_values[-1]))
        self.logger.info('Portfolio Balance:      ${:.2f}'.format(self.balance))
        self.logger.info('Portfolio Stocks Number: {}'.format(len(self.inventory)))
        self.logger.info('Total Return:           ${:.2f}'.format(portfolio_return))
        self.logger.info('Mean/Daily Return Rate:  {:.3f}%'.format(np.mean(self.return_rates) * 100))
        # self.logger.info('Sharpe Ratio adjusted with Treasury bond daily return: {:.3f}'.format(sharpe_ratio(np.array(agent.return_rates)), risk_free=treasury_bond_daily_return_rate()))
        # self.logger.info('Maximum Drawdown:        {:.3f}%'.format(maximum_drawdown(agent.portfolio_values) * 100))
        self.logger.info("--------------------------------")
        
        return portfolio_return
    
    def plot_portfolio(self):
        
        '''combined plots of portfolio transaction history and performance comparison'''
        
        fig, ax = plt.subplots(2, 1, figsize=(16,8), dpi=100)

        fig.autofmt_xdate()

        portfolio_return = self.portfolio_values[-1] - self.initial_portfolio_value

        # Load the entire dataset from year 7 onwards for 3 years
        df = self.env.filter_data(7, self.train_period)
        # df.drop("Return", inplace=True)
        df = df * self.env.std + self.env.mean

        buy_prices = [df.iloc[t, 4] for t in self.buy_dates]
        sell_prices = [df.iloc[t, 4] for t in self.sell_dates]

        # Portfolio transaction chart
        ax[0].set_title('GDQN Total Return: ${:.2f}'.format(portfolio_return))
        ax[0].plot(df.index, df['Adj Close'], color='black', label="GOOG") # stock name hardcoded for now
        ax[0].scatter(df.index[self.buy_dates], buy_prices, c='green', alpha=0.5, label='buy')
        ax[0].scatter(df.index[self.sell_dates], sell_prices,c='red', alpha=0.5, label='sell')
        ax[0].set_ylabel('Price')
        ax[0].set_xticks(df.index[::len(df)//10])
        ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax[0].xaxis.set_major_locator(mdates.MonthLocator())
        ax[0].tick_params(axis='x', rotation=45)
        ax[0].legend()
        ax[0].grid()

        # Comparison to benchmark
        dates, buy_and_hold_portfolio_values, buy_and_hold_return = buy_and_hold_benchmark(self)
        ax[1].set_title('GDQN vs. Buy and Hold')
        # This line has some issues to be fixed
        # ax[1].plot(dates, self.portfolio_values, color='green', label='GDQN Total Return: ${:.2f}'.format(portfolio_return))
        ax[1].plot(dates, buy_and_hold_portfolio_values, color='blue', label='Buy and Hold Total Return: ${:.2f}'.format(buy_and_hold_return))
        ax[1].set_ylabel('Portfolio Value ($)')
        ax[1].set_xticks(df.index[::len(df)//10])
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax[1].xaxis.set_major_locator(mdates.MonthLocator())
        ax[1].tick_params(axis='x', rotation=45)
        ax[1].legend()
        ax[1].grid()

        plt.subplots_adjust(hspace=0.5)
        plt.show()

def buy_and_hold_benchmark(self):
    
    df = self.env.filter_data(7, self.train_period)
    # df.drop("Return", inplace=True)
    df = df * self.env.std + self.env.mean
    dates = df.index.values

    num_holding = self.initial_portfolio_value // df.iloc[0, 5]
    balance_left = self.initial_portfolio_value % df.iloc[0, 5]
    
    buy_and_hold_portfolio_values = df['Adj Close']*num_holding + balance_left
    buy_and_hold_return = buy_and_hold_portfolio_values.iloc[-1] - self.initial_portfolio_value
    
    return dates, buy_and_hold_portfolio_values, buy_and_hold_return

print("GDQN Agent imported successfully for testing!")