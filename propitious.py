import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Propitious:
    def __init__(self, w, alpha1, alpha2, L1, L2, beta=0.8, eps=0.8):
        self.w = w
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.L1 = L1
        self.L2 = L2
        self.beta = beta
        self.eps = eps
        
    def plot_alphas(self):
        plt.scatter(self.alpha1, self.alpha2, alpha=0.1)
        plt.xlabel(r'$\alpha_1$')
        plt.ylabel(r'$\alpha_2$')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.grid()
        plt.show()
        
    def create_figure(self, P):
        self.calc_UP_A1_A2(P)
        alpha1pref = [alpha1 for alpha1 in np.linspace(0, self.UP/self.A1, 100)]
        alpha2pref = [self.UP/self.A2 - alpha1 * self.A1/self.A2
                      for alpha1 in alpha1pref]
        
        C = self.average_cost(P)
        alpha1cost = [alpha1 for alpha1 in np.linspace(0, C/self.L1, 100)]
        alpha2cost = [C/self.L2 - alpha1 * self.L1/self.L2
                      for alpha1 in alpha1cost]
        
        plt.scatter(self.alpha1, self.alpha2, alpha=0.1)
        plt.plot(alpha1pref, alpha2pref, label='isopreference')
        plt.plot(alpha1cost, alpha2cost, label='isocost')
        plt.xlabel(r'$\alpha_1$')
        plt.ylabel(r'$\alpha_2$')
        plt.grid()
        plt.legend()
        plt.show()
        
        
    def utility(self, x):
        return x**(1 - self.eps) / (1 - self.eps)
        
    def calc_UP_A1_A2(self, P):
        self.UP = self.beta * (self.utility(self.w) - self.utility(self.w - P))
        self.A1 = self.utility(self.w - P) - self.utility(self.w - self.L1) + self.UP
        self.A2 = self.utility(self.w - P) - self.utility(self.w - self.L2) + self.UP

    def participation(self, P):
        self.calc_UP_A1_A2(P)
        return self.alpha1 * self.A1 + self.alpha2 * self.A2 - self.UP > 0
        
    def average_cost(self, P):
        insured = self.participation(P)
        N = sum(insured * (self.alpha1 * self.L1 + self.alpha2 * self.L2))
        D = sum(insured)
        return N / (D + 1e-16)
    
    def check_propitious_selection(self, price_change=1000):
        prices = np.arange(0, self.L1+1, price_change)
        average_costs = []
        old_cost = 0
        for price in prices:
            new_cost = self.average_cost(price)
            average_costs.append(new_cost)
            if new_cost < 1:
                return False
            if new_cost < old_cost:
                return True
            old_cost = new_cost
    
    def find_prices_propitious_selection(self, step=100):
        prices = np.arange(0, self.L1+1, step)
        old_cost = 0
        prices_propitious = []
        price_star, max_profit = 0, 0
        for price in prices:
            new_cost = self.average_cost(price)
            if new_cost < 1:
                return price_star, prices_propitious
            if new_cost < old_cost:
                prices_propitious.append(price - step)
            old_cost = new_cost
            profit = sum(self.participation(price)) * (price - new_cost)
            if profit > max_profit:
                max_profit = profit
                price_star = price
                
    def find_prices_profits_propitious_selection(self, step=100):
        prices = np.arange(0, self.L1+1, step)
        old_cost = 0
        prices_propitious = []
        profits_propitious = []
        price_star, max_profit = 0, 0
        for price in prices:
            new_cost = self.average_cost(price)
            if new_cost < 1:
                df = pd.DataFrame({'price': prices_propitious,
                                   'profit': profits_propitious})
                df['remark'] = 'propitious selection'
                df = df.append({'price': price_star,
                                'profit': max_profit,
                                'remark': 'max profit'}, ignore_index=True)
                df.sort_values('price', inplace=True)
                return df.round()
            if new_cost < old_cost:
                price_propitious = price - step
                profit_propitious = (sum(self.participation(price_propitious))
                                     * (price_propitious - new_cost))
                prices_propitious.append(price_propitious)
                profits_propitious.append(profit_propitious)
            old_cost = new_cost
            profit = sum(self.participation(price)) * (price - new_cost)
            if profit > max_profit:
                max_profit = profit
                price_star = price
            
    def plot_average_cost_and_profit(self, step=100):
        prices = np.arange(0, self.L1+1, step)
        average_costs = []
        profits = []
        old_cost = 0
        for price in prices:
            new_cost = self.average_cost(price)
            average_costs.append(new_cost)
            profit = sum(self.participation(price)) * (price - new_cost)
            profits.append(profit)
            if new_cost < 1:
                break
        fig, ax1 = plt.subplots()
        
        color = 'tab:red'
        ax1.set_xlabel('price')
        ax1.set_ylabel('average_cost', color=color)
        ax1.plot(prices[:len(average_costs)], average_costs, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx() 
        color = 'tab:blue'
        ax2.set_ylabel('profit', color=color)
        ax2.plot(prices[:len(profits)], profits, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.grid()

        fig.tight_layout()
        plt.show()
        
def plot_propitious_selection(l_propitious):
    L1s = [L1 for L1, L2 in l_propitious]
    L2s = [L2 for L1, L2 in l_propitious]

    plt.scatter(L1s, L2s)
    plt.xlabel('L1')
    plt.ylabel('L2')
    plt.xlim([0, 100e3])
    plt.ylim([0, 100e3])
    plt.grid()
    plt.show()