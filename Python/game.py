import numpy.random as npr
import random

from Agents.random_agent import RandomAgent
from HelperClasses import Memory
import constants as c
from HelperFunctions import *

# initialize agents
# can update with inclusion of informed or noise traders
agents = []
for n in range(c.NUM_AGENTS):
	agents.append( RandomAgent() )

# initialize variables for game loop
n = 0 # count number of rounds
if not c.CONCURRENT_DIVIDEND:
	dividend_out = True # track whether next dividend date is determined

	# determine dividend date and size
	dividend_time = n + npr.geometric(p=c.DIVIDEND_PROB)
	dividend_amt = npr.normal(c.DIVIDEND_MEAN, c.DIVIDEND_STD)

	# select which agents know about dividend
	informed_agents = random.sample(agents, npr.randint( len(agents) ))
	for a in informed_agents:
		a.update_info(dividend_time, dividend_amt)

# initialize market
current_market = {'bid prices': None,
				  'bid volumes': None,
				  'ask prices': None,
				  'ask volumes': None,
				  'round': n}
buy_actions = {} # used to organize actions from agents and to create market
sell_actions = {}

mem = Memory(mem_size=100000, batch_size=c.BATCH_SIZE, window=c.BATCH_WINDOW)

# run game loop
while True:
	if n > c.DEBUG_ROUNDS:
		break
	# get actions from agents
	# actions = []
	for a in agents:
		if n < c.BATCH_SIZE + c.BATCH_WINDOW:
			act = a.act()
			p = act['price']
			v = act['volume']
			if act['action'] == c.BUY_ORDER: 	# action is buy order
				if p in sell_actions: 			# buy order fills current ask
					ask_v = sell_actions[p]
					if ask_v > v:
						sell_actions[p] -= v 	# update price-level volume
						break
					else:
						del sell_actions[p]
				if p in buy_actions: 			# buy order price level exists, update
					buy_actions[p] += v
				else: 							# buy order price level doesn't currently exist
					buy_actions[p] = v
			if act['action'] == c.SELL_ORDER: 	# action is sell order
				if p in buy_actions:			# sell order fills current bid
					bid_v = buy_actions[p]
					if bid_v > v:
						buy_actions[p] -= v 	# update pricel-level volume
						break
					else:
						del buy_actions[p]
				if p in sell_actions:			# sell order price level exists, update
					sell_actions[p] += v
				else:
					sell_actions[p] = v 		# buy order price level doesn't currently exist
			# actions.append(a.act())
		else:
			current_state = mem.get_current_state()
			actions.append(a.act(current_state))


	# organize buy/sell orders in descending/ascending
	# buy_orders = sorted(filter(lambda action: action['action'] == c.BUY_ORDER, actions), key=lambda action: action['price'], reverse=True)
	# sell_orders = sorted(filter(lambda action: action['action'] == c.SELL_ORDER, actions), key=lambda action: action['price'])
	bid_px, bid_vol = dict_to_list(buy_actions, bid=True)
	ask_px, ask_vol = dict_to_list(sell_actions, bid=False)

	# fill orders
	fill_orders(bid_px, bid_vol, ask_px, ask_vol)

	# create new market state
	current_market = {'bid prices': bid_px,
					  'bid volumes': bid_vol,
					  'ask prices': ask_px,
					  'ask volumes': ask_vol,
					  'round': n}

	# update memory
	mem.add_memory(current_market)

	# distribute dividend (could have it at beginning of period?)
	if n == dividend_time:
		for a in informed_agents:
			a.update_val(dividend_amt)
			a.update_info(next_dividend=-1, dividend_amt=0)
		dividend_out = False

	# select next dividends if none and which agents know
	if not dividend_out:
		dividend_out = True # track whether next dividend date is determined

		# determine dividend date and size
		dividend_time = n + npr.geometric(p=c.DIVIDEND_PROB)
		dividend_amt = npr.normal(c.DIVIDEND_MEAN, c.DIVIDEND_STD)

		# select which agents know about dividend
		informed_agents = random.sample(agents, npr.randint( len(agents) ))
		for a in informed_agents:
			a.update_info(dividend_time, dividend_amt)

	n += 1

	if c.DEBUG:
		print(current_market)
