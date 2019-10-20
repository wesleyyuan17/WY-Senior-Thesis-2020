import numpy as np
import random

from Agents.random_agent import RandomAgent
from HelperClasses import *
import constants as c
from HelperFunctions import *

# initialize agents
# can update with inclusion of informed or noise traders
training_agents = {}
for n in range(c.NUM_TRAINED_AGENTS):
	training_agents['t' + str(n)] = RandomAgent(c.NOISE_ACTION_SPACE, n)

noise_agents = {}
for n in range(c.NUM_NOISE_AGENTS):
	noise_agents['n' + str(n)] = RandomAgent(c.NOISE_ACTION_SPACE, n)

# initialize variables for game loop
n = 0 				# count number of rounds
dividend_out = True # track whether next dividend date is determined

# determine dividend date and size
dividend_time = n + np.random.geometric(p=c.DIVIDEND_PROB)
dividend_amt = np.random.normal(c.DIVIDEND_MEAN, c.DIVIDEND_STD)

# select which agents know about first dividend
##################
# Needs work lol #
##################
informed_agents = random.sample(list(training_agents.values()), int(c.NUM_TRAINED_AGENTS * c.PERCENT_INFORMED))
for a in informed_agents:
	a.update_info(dividend_time, dividend_amt)

# initialize market
current_market = np.zeros(shape=(4,10)) # each row is top 10 of bid/ask price/volume
agent_actions = {}
# buy_actions = {} # key is price, value is list of tuples of (agent ID, orderID, cancellation times)
# sell_actions = {}
# min_ask = np.inf
# max_bid = -np.inf

# create memory object
###################################
# memory object needs refactoring #
###################################
mem = Memory(mem_size=100000, batch_size=c.BATCH_SIZE, window=c.BATCH_WINDOW)
market = Market()

# run game loop
while True:
	if c.DEBUG:
		print(n)
		print(current_market)

	if n > c.DEBUG_ROUNDS:
		break

	# get actions from agents
	for agId, ag in noise_agents.items():
		agent_actions[agId] = ag.act(current_market, n)
	for agId, ag in training_agents.items():
		agent_actions[agId] = ag.act(current_market, n)

	# pass actions to market object to update state and agents
	current_market = market.update(agent_actions, training_agents, noise_agents, n)

	# update memory
	mem.add_memory(current_market)

	# distribute dividend (could have it at beginning of period?)
	if n == dividend_time:
		for _, a in training_agents.items():
			a.update_val(dividend_amt)
			a.update_info(next_dividend=-1, dividend_amt=0)
		for _, a in noise_agents.items():
			a.update_val(dividend_amt)
			a.update_info(next_dividend=-1, dividend_amt=0)
		dividend_out = False

	# select next dividends if none and which agents know
	if not dividend_out:
		dividend_out = True # track whether next dividend date is determined

		# determine dividend date and size
		dividend_time = n + np.random.geometric(p=c.DIVIDEND_PROB)
		dividend_amt = np.random.normal(c.DIVIDEND_MEAN, c.DIVIDEND_STD)

		# select which agents know about dividend
		informed_agents = random.sample(list(training_agents.values()), int(c.NUM_TRAINED_AGENTS * c.PERCENT_INFORMED))
		for a in informed_agents:
			a.update_info(dividend_time, dividend_amt)

	n += 1
