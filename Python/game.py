import numpy.random as npr
import random

from Agents.random_agent import RandomAgent
from HelperClasses import Memory
import constants as c

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
current_market = {'bid': None,
				  'bid depth': None,
				  'ask': None,
				  'ask depth': None,
				  'round': n}

mem = Memory(mem_size=100000, batch_size=c.BATCH_SIZE, window=c.BATCH_WINDOW)

# run game loop
while True:
	if n > c.DEBUG_ROUNDS:
		break
	# get actions from agents
	actions = []
	for a in agents:
		if n < c.BATCH_SIZE + c.BATCH_WINDOW:
			actions.append(a.act())
		else:
			current_state = mem.get_current_state()
			actions.append(a.act(current_state))


	# organize buy/sell orders in descending/ascending
	buy_orders = sorted(filter(lambda action: action['action'] == c.BUY_ORDER, actions), key=lambda action: action['price'], reverse=True)
	sell_orders = sorted(filter(lambda action: action['action'] == c.SELL_ORDER, actions), key=lambda action: action['price'])

	# fill orders
	while len(buy_orders) > 0 and len(sell_orders) > 0 and buy_orders[0]['price'] >= sell_orders[0]['price']:
		buy_volume = buy_orders[0]['volume']
		sell_volume = sell_orders[0]['volume']
		if buy_volume < sell_volume:
			sell_orders[0]['volume'] = sell_volume - buy_volume
			del buy_orders[0]
		elif buy_volume > sell_volume:
			buy_orders[0]['volume'] = buy_volume - sell_volume
			del sell_orders[0]
		else:
			del buy_orders[0]
			del sell_orders[0]

	if len(sell_orders) == 0:

		bid_depth = 0
		for b in buy_orders:
			bid_depth += b['volume']

		current_market = {'bid': buy_orders[0]['price'],
						  'bid depth': buy_orders[0]['volume'], # top volume, maybe change to sum of volumes?
						  'ask': None,
						  'ask depth': 0,
						  'round': n}
	elif len(buy_orders) == 0:

		ask_depth = 0
		for s in sell_orders:
			ask_depth += s['volume']

		current_market = {'bid': None,
						  'bid depth': 0, # top volume, maybe change to sum of volumes?
						  'ask': sell_orders[0]['price'],
						  'ask depth': ask_depth,
						  'round': n}
	else:

		bid_depth = 0
		for b in buy_orders:
			bid_depth += b['volume']

		ask_depth = 0
		for s in sell_orders:
			ask_depth += s['volume']

		current_market = {'bid': buy_orders[0]['price'],
						  'bid depth': bid_depth, # top volume, maybe change to sum of volumes?
						  'ask': sell_orders[0]['price'],
						  'ask depth': ask_depth,
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
