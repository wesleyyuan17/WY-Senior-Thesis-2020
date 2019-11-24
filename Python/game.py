import numpy as np
import random

from Agents.random_agent import RandomAgent
from Agents.simple_agent import SimpleAgent
from MarketClass import *
from constants import *

if DEBUG:
	import csv
	import matplotlib.pyplot as plt

# initialize agents
# can update with inclusion of informed or noise traders
training_agents = {}
for n in range(NUM_TRAINED_AGENTS):
	training_agents['t' + str(n)] = SimpleAgent(ID=n, state_size=MARKET_DEPTH, informed=True)

noise_agents = {}
for n in range(NUM_NOISE_AGENTS):
	noise_agents['n' + str(n)] = RandomAgent(n_actions=NOISE_ACTION_SPACE, ID=n)

# initialize variables for game loop
n = 0 				# count number of rounds
dividend_out = True # track whether next dividend date is determined

# determine dividend date and size
dividend_time = n + np.random.geometric(p=DIVIDEND_PROB)
dividend_amt = np.random.normal(DIVIDEND_MEAN, DIVIDEND_STD)

# select which agents know about first dividend
##################
# Needs work lol #
##################
informed_agents = random.sample(list(training_agents.values()), int(NUM_TRAINED_AGENTS * PERCENT_INFORMED))
for a in informed_agents:
	a.update_info(dividend_time, dividend_amt)

# initialize market
current_market = np.zeros(shape=(4,MARKET_DEPTH)) # each row is top 10 of bid/ask price/volume
agent_actions = {}
# buy_actions = {} # key is price, value is list of tuples of (agent ID, orderID, cancellation times)
# sell_actions = {}
# min_ask = np.inf
# max_bid = -np.inf

# create memory object
###################################
# memory object needs refactoring #
###################################
# mem = Memory(mem_size=100000, batch_size=BATCH_SIZE, window=BATCH_WINDOW)
market = Market(MARKET_DEPTH)

# run game loop
price = 0
num_trades_filled = 0
prices = []
while True:
	if DEBUG:
		# print(n)
		# print('current market:\n', current_market)
		# print('price:', price)
		prices.append(price)

	if n > DEBUG_ROUNDS:
		# write to csv for reading in data in R for statistical test
		with open('prices.csv', 'w', newline='') as myfile: 
			wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
			i = 0
			for p in prices:
				if i % 100 == 0:
					wr.writerow([p])
				i += 1

		# plot graph of price movements
		plt.figure(figsize=(50,10))
		plt.plot(range(0,len(prices)), prices)
		break
		'''
		Tests show prices are somewhat realistic with Dickey-Fuller tests showing log-prices move with high probability
		following a random walk with drift
		Drift: -135.0677 (-8.437 critical value)
		No Drift: 3602.554 (-13.96 critical value)
		'''

	# get actions from agents
	for agId, ag in noise_agents.items():
		agent_actions[agId] = ag.act(current_market, n)
	for agId, ag in training_agents.items():
		ag.act(current_market, price, num_trades_filled, n)

	# pass actions to market object to update state and agents
	current_market, price, num_trades_filled = market.update(agent_actions, training_agents, noise_agents, n)

	# update memory
	# mem.add_memory(current_market)

	# distribute dividend (could have it at beginning of period?)
	if n == dividend_time:
		for _, a in training_agents.items():
			a.update_val(dividend_amt)
			# a.update_info(next_dividend=-1, dividend_amt=0)
		for _, a in noise_agents.items():
			a.update_val(dividend_amt)
			# a.update_info(next_dividend=-1, dividend_amt=0)
		dividend_out = False

	# select next dividends if none and which agents know
	if not dividend_out:
		dividend_out = True # track whether next dividend date is determined

		# determine dividend date and size
		dividend_time = n + np.random.geometric(p=DIVIDEND_PROB)
		dividend_amt = np.random.normal(DIVIDEND_MEAN, DIVIDEND_STD)

		# select which agents know about dividend
		informed_agents = random.sample(list(training_agents.values()), int(NUM_TRAINED_AGENTS * PERCENT_INFORMED))
		for a in informed_agents:
			a.update_info(dividend_time, dividend_amt)

	n += 1

plt.show()
