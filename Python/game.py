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

# initialize agents
# can update with inclusion of informed or noise traders
informed_agents = {}
for n in range(NUM_TRAINED_AGENTS):
	informed_agents['i' + str(n)] = SimpleAgent(ID=n, state_size=MARKET_DEPTH, informed=True, eval=False)

### no uninformed traders yet ###
uninformed_agents = {}
for n in range(NUM_TRAINED_AGENTS):
	uninformed_agents['u' + str(n)] = SimpleAgent(ID=n, state_size=MARKET_DEPTH, informed=False, eval=False)

training_agents = {**informed_agents, **uninformed_agents}

noise_agents = {}
for n in range(NUM_NOISE_AGENTS):
	noise_agents['n' + str(n)] = RandomAgent(n_actions=NOISE_ACTION_SPACE, ID=n)

# initialize variables for game loop
n = 1 							# count number of rounds
dividend_out = True 			# track whether next dividend date is determined
price = 0						# last market-clearing price
num_trades_filled = 0			# number of trades executed at last price
informed_agents_pnl = 0		# do the informed agents make money?
market = Market(MARKET_DEPTH) 	# market object that clears trades and maintains LOB

# determine dividend date and size
dividend_time = n + 10 # np.random.geometric(p=DIVIDEND_PROB)
dividend_amt = np.random.choice([-20,20]) # np.random.normal(DIVIDEND_MEAN, DIVIDEND_STD)

# give informed traders information
for _, a in informed_agents.items():
	a.update_info(dividend_time, dividend_amt)

# initialize market
current_market = np.zeros(shape=(4,MARKET_DEPTH)) # each row is top 10 of bid/ask price/volume
agent_actions = {}

# for writing out results
f_prices = open('prices_aws.csv', 'w', newline='')
wr_prices = csv.writer(f_prices)

f_pnl = open('informed_agents_pnl_aws.csv', 'w', newline='')
wr_pnl = csv.writer(f_pnl)

f_loss = open('informed_training_loss_aws.csv', 'w', newline='')
wr_loss = csv.writer(f_loss)

fu_pnl = open('uninformed_agents_pnl_aws.csv', 'w', newline='')
wru_pnl = csv.writer(fu_pnl)

fu_loss = open('uninformed_training_loss_aws.csv', 'w', newline='')
wru_loss = csv.writer(fu_loss)

# run game loop
while True:
	if DEBUG:
		print('\ntime step:', n)
		# print('current market:\n', current_market)
		# print('price:', price)
		wr_prices.writerow([price])

		# write informed agents data
		informed_agents_pnl = informed_agents['i0'].current_value()
		wr_pnl.writerow([informed_agents_pnl])

		wr_loss.writerow([informed_agents['i0'].training_loss()])

		# write uninformed agents data
		uninformed_agents_pnl = uninformed_agents['u0'].current_value()
		wru_pnl.writerow([uninformed_agents_pnl])

		wru_loss.writerow([uninformed_agents['u0'].training_loss()])

	if n > DEBUG_ROUNDS:
		break

	# get actions from agents
	for aId, a in noise_agents.items():
		agent_actions[aId] = a.act(current_market, n)
	for aId, a in training_agents.items():
		agent_actions[aId] = a.act(current_market, price, num_trades_filled, n)
		
	# pass actions to market object to update state and agents
	current_market, price, num_trades_filled = market.update(agent_actions, training_agents, noise_agents, n)

	# distribute dividend (could have it at beginning of period?)
	if n == dividend_time:
		for _, a in training_agents.items():
			a.update_val(dividend_amt)
		for _, a in noise_agents.items():
			a.update_val(dividend_amt)
		dividend_out = False

	# select next dividends if none and which agents know
	if not dividend_out:
		dividend_out = True # track whether next dividend date is determined

		# determine dividend date and size
		dividend_time = n + 10 # np.random.geometric(p=DIVIDEND_PROB)
		dividend_amt = np.random.choice([-20,20]) # np.random.normal(DIVIDEND_MEAN, DIVIDEND_STD)

		# notify agents about dividend info
		for _, a in informed_agents.items():
			a.update_info(dividend_time, dividend_amt)

	n += 1

### after loop, save trained DQNs ###
for _, a in training_agents.items():
	a.save_DQN()
