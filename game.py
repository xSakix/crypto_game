import pandas as pd
import numpy as np
import os


try:
    import msvcrt
    getch = msvcrt.getch
except:
    import sys, tty, termios
    def _unix_getch():
        """Get a single character from stdin, Unix version"""

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())          # Raw read
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    getch = _unix_getch

class CryptoGameAgent:
    def __init__(self, ticket, use_trader=False, invested=100000., model=None, agent=None, r_actions=np.empty(0)):
        self.r_actions = r_actions
        self.score = 0
        self.ror_history = []
        self.dist = np.array([1.])
        self.ticket = ticket
        self.tr_cost = 0.003
        self.invested = invested
        self.cash = invested
        self.shares = np.zeros(1)
        self.history = []
        self.actions = []
        self.state = []
        self.coef = 1.
        self.use_trader = use_trader
        self.model = model
        self.agent = agent

    def invest(self,
               data,
               window,
               pct,
               bench,
               mean,
               median,
               lowerbb,
               upperbb):

        if len(data.keys()) == 0:
            return

        if len(self.r_actions) == 0:
            self.r_actions = np.random.randint(0, 3, size=len(data) - window)

        self.ror_history = np.empty(len(pct))
        self.ror_history[:] = np.nan
        os.system('cls')

        for i in range(window, len(data)):
            input = [
                pct[i],
                lowerbb[i],
                mean[i],
                median[i],
                upperbb[i]]
            cur_state = str(input).replace('array', '') \
                .replace('(', '') \
                .replace(')', '') \
                .replace('[', '') \
                .replace(']', '') \
                .replace(',', ' |')
            print('\t\t\t PCT\t\t      |\tLOWER\t    |\tMEAN\t  |\tMEDIAN\t|\tUPPER')
            print('\n%d/%d: Current state is: %s' % (i, len(data), cur_state))
            action = self.get_action(i - window)
            os.system('cls')
            prices = data.iloc[i]
            portfolio = self.cash + np.dot(prices, self.shares)

            try:
                if np.isnan(portfolio):
                    portfolio = 0.
            except:
                print('portfolio:', portfolio)

            self.history.append(portfolio)

            self.ror_history[i] = (portfolio - self.invested) / self.invested
            if self.use_trader and i > window:
                # print('%.3f : %.3f : %.3f' % (ror, self.agent.ror_history[i], bench[i]))
                self.score_based_on_beating_trader(self.ror_history[i], self.agent.ror_history[i])
            elif i > window:
                # self.score_based_on_ror(ror)
                self.score_based_on_beating_benchmark(self.ror_history[i], bench[i])

            self.state.append(np.array(input))

            self.do_action(action, portfolio, prices)

        df = pd.DataFrame(self.ror_history)
        df.fillna(method="bfill", inplace=True)
        self.ror_history = df.as_matrix()
        self.state = np.array(self.state)

    def do_action(self, action, portfolio, prices):
        if action == 1 and sum(self.shares > 0) > 0:
            to_sell = self.coef * self.shares
            sold = np.dot(to_sell, prices)
            self.cash += sold - sold * self.tr_cost
            self.shares = self.shares - to_sell
            # portfolio = self.cash + np.dot(prices, self.shares)
            # print('selling ', to_sell, ' portfolio=', portfolio, 'cash=', self.cash,'shares=',self.shares)
        elif action == 2 and (self.coef * self.cash - self.tr_cost * self.coef * self.cash) > 0.000000001 * prices:
            c = self.cash * self.coef
            cost = np.multiply(self.tr_cost, c)
            c = np.subtract(c, cost)
            s = np.divide(c, prices)
            self.shares += s
            self.cash = portfolio - np.dot(self.shares, prices) - cost
            # portfolio = self.cash + np.dot(prices, self.shares)
            # print('buying ', s, ' portfolio=', portfolio, 'cash=', self.cash,'shares=',self.shares)

    def get_action(self, i):
        print('\rEnter action(0-Hold, 1-SELL,2-BUY): ',end='')
        return int(getch())
        # return int(input("Enter action(0-Hold, 1-SELL,2-BUY): "))

    def score_based_on_ror(self, ror):
        if ror > 0.:
            self.score += 1

    def score_based_on_beating_benchmark(self, ror, benchmark, leverage=1.0):
        if ror * leverage >= benchmark:
            self.score += 1

    def score_based_on_beating_trader(self, ror, ror_agent):
        if ror >= ror_agent:
            self.score += 1


start_date = '2018-05-01'
end_date = '2018-07-01'
ticket = 'BTC-EUR'

data = pd.read_csv('crypto_data_adj_close.csv')
data = data[data['date'] >= start_date]
data = data[data['date'] <= end_date]
print(data.head())
print(data.tail())

data = data[ticket]

window = 30

pct = data.pct_change().as_matrix()
bench = data.pct_change().cumsum().as_matrix()
data_1 = pd.DataFrame(pct)
mean = data_1.rolling(window=window).mean().as_matrix()
median = data_1.rolling(window=window).median().as_matrix()
std = data_1.rolling(window=window).std().as_matrix()
upperbb = mean + (2 * std)
lowerbb = mean - (2 * std)

agent = CryptoGameAgent(ticket)
agent.invest(data, window, pct, bench, mean, median, lowerbb, upperbb)

print('RESULT:')
print('score:', agent.score)
print('ror', agent.ror_history[-1])
print('portfolio/invested = %f/%f' % (agent.history[-1], agent.invested))
