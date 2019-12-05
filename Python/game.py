import numpy as np
import random
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from Agents.random_agent import RandomAgent
from Agents.simple_agent import SimpleAgent
from MarketClass import *
from constants import *

if DEBUG:
	import csv
	import matplotlib.pyplot as plt

# initialize agents
# can update with inclusion of informed or noise traders
informed_agents = {}
for n in range(NUM_TRAINED_AGENTS):
	informed_agents['i' + str(n)] = SimpleAgent(ID=n, state_size=MARKET_DEPTH, informed=True)

### no uninformed traders yet ###
uninformed_agents = {}
# for n in range(NUM_TRAINED_AGENTS):
# 	uninformed_agents['u' + str(n)] = SimpleAgent(ID=n, state_size=MARKET_DEPTH, informed=False)

training_agents = {**informed_agents, **uninformed_agents}

noise_agents = {}
for n in range(NUM_NOISE_AGENTS):
	noise_agents['n' + str(n)] = RandomAgent(n_actions=NOISE_ACTION_SPACE, ID=n)

# initialize variables for game loop
n = 1 							# count number of rounds
dividend_out = True 			# track whether next dividend date is determined
price = 0						# last market-clearing price
num_trades_filled = 0			# number of trades executed at last price
prices = []						# for visualizing price movements
informed_agents_value = []		# do the informed agents make money?
market = Market(MARKET_DEPTH) 	# market object that clears trades and maintains LOB

# determine dividend date and size
dividend_time = n + np.random.geometric(p=DIVIDEND_PROB)
dividend_amt = np.random.normal(DIVIDEND_MEAN, DIVIDEND_STD)

# select which agents know about first dividend - change to being constant subset that are always informed
# informed_agents = random.sample(list(training_agents.values()), int(NUM_TRAINED_AGENTS * PERCENT_INFORMED))

# give informed traders information
for _, a in informed_agents.items():
	a.update_info(dividend_time, dividend_amt)

# initialize market
current_market = np.zeros(shape=(4,MARKET_DEPTH)) # each row is top 10 of bid/ask price/volume
agent_actions = {}
# buy_actions = {} # key is price, value is list of tuples of (agent ID, orderID, cancellation times)
# sell_actions = {}
# min_ask = np.inf
# max_bid = -np.inf

# run game loop
while True:
	if DEBUG:
		print('time step:', n)
		print('current market:\n', current_market)
		print('price:', price)
		prices.append(price)
		informed_agents_value.append(informed_agents['i0'].current_value())

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
		plt.title('Price Movements')

		# plot portfolio value of informed agent
		plt.figure(figsize=(50,10))
		plt.plot(range(0, len(informed_agents_value)), informed_agents_value)
		plt.title('Informed Agent Portfolio Value')

		# plot loss
		loss = informed_agents['i0'].training_loss()
		plt.figure(figsize=(50,10))
		plt.plot(range(0, len(loss)), loss)
		plt.title('Training Loss')

		break

	# get actions from agents
	for aId, a in noise_agents.items():
		agent_actions[aId] = a.act(current_market, n)
	for aId, a in training_agents.items():
		agent_actions[aId] = a.act(current_market, price, num_trades_filled, n)

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
		# informed_agents = random.sample(list(training_agents.values()), int(NUM_TRAINED_AGENTS * PERCENT_INFORMED))
		for _, a in informed_agents.items():
			a.update_info(dividend_time, dividend_amt)

	n += 1

for a in informed_agents.items():
	a.save_DQN()
plt.show()
