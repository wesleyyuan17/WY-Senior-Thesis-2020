import numpy as np

class Market():
	'''
	Market object that handles creation and maintenance of limit order book
	'''
	buy_actions = None
	sell_actions = None
	min_ask = None
	max_bid = None
	current_market = None
	n = None
	k = None

	def __init__(self, k):
		'''
		Initializes market object
		Args:
			k: int, depth of market state desired as return
		'''
		self.buy_actions = {} 								# dict, key is price, value is list of tuples of (agentId, orderId, duration)
		self.sell_actions = {} 								# dict, key is price, value is list of tuples of (agentId, orderId, duration)
		self.price = 0 										# market-clearing price
		self.num_trades_filled = 0 							# number of orders filled at market-clearing price
		self.k = k 											# depth of market ot return
		self.current_market = np.zeros(shape=(4,self.k)) 	# current market environment as 4xk matrix

	def update(self, agent_actions, training_agents, noise_agents, n):
		'''
		called in game to perform necessary updates to market and agents
		Args:
			agent_actions: dict, key is agentId, value is action for this round for agent
			training_agents: dict, key is agentId, value is the agent object
			noise_agents: dict, key is agentId, value is the agent object
			n: int, round number of game
		Returns:
			a 4xk numpy array of bid price/volume then ask price/volume as rows to depth k
		'''
		self.clear_market(agent_actions, training_agents, noise_agents)

		self.cancel_orders(n)
		self.create_market()

		return self.current_market, self.price, self.num_trades_filled

	def clear_market(self, agent_actions, training_agents, noise_agents):
		'''
		iterates through actions and adds them to LOB, then finds market-clearing price and fills
		orders and notifies agents
		Args:
			agent_actions: dict, key is agentId, value is action for this round for agent
			training_agents: dict, key is agentId, value is the agent object
			noise_agents: dict, key is agentId, value is the agent object
		'''

		# add orders to limit order book
		for agId, act in agent_actions.items():
			self.add_to_book(agId, act)

		# find the market-clearing price
		bid_px_vol = self.price_volume(bid=True)
		ask_px_vol = self.price_volume(bid=False)

		bid_px = bid_px_vol[:,0] # all prices of bid orders currently in order book
		bid_vol = bid_px_vol[:,1] # corresponding volumes for prices currently in order book
		ask_px = ask_px_vol[:,0] # all prices of ask orders currently in order book
		ask_vol = ask_px_vol[:,1] # corresponding volumes for prices currently in order book
		bid_idx = 0 # track price of bids
		ask_idx = 0 # track price of sells
		num_bids_filled = 0 # track total bid volume
		num_asks_filled = 0 # track total ask volume
		while bid_px[bid_idx] >= ask_px[ask_idx]:
			temp_bid_vol = bid_vol[bid_idx]
			temp_ask_vol = ask_vol[ask_idx]
			# at this price level, more bids than ask - move to next ask price level, increment ask
			if num_bids_filled + temp_bid_vol > num_asks_filled + temp_ask_vol:
				num_asks_filled += temp_ask_vol
				ask_idx += 1
			# at this price level, more asks than bids - move to next ask price level, increment ask
			elif num_bids_filled + temp_bid_vol < num_asks_filled + temp_ask_vol:
				num_bids_filled += temp_bid_vol
				bid_idx += 1
			# at this price level, voluems are equal - increment both to next price level
			else:
				num_asks_filled += temp_ask_vol
				ask_idx += 1
				num_bids_filled += temp_bid_vol
				bid_idx += 1

		self.num_trades_filled = int(max(num_bids_filled, num_asks_filled)) # trades filled is minimum crossing volume
		if num_bids_filled > num_asks_filled: # more bids than asks, orders filled at highest ask level
			self.price = ask_px[ask_idx]
		elif num_bids_filled < num_asks_filled: # more asks than bids, orders filled at lowest bid level
			self.price = bid_px[bid_idx]
		else: # volumes equal, orders filled at mid
			self.price = 0.5 * (ask_px[ask_idx] + bid_px[bid_idx])

		# fill orders
		self.fill_orders(bid_px, bid_vol, bid_idx, training_agents, noise_agents, True)
		self.fill_orders(ask_px, ask_vol, ask_idx, training_agents, noise_agents, False)

	def fill_orders(self, px, vol, idx, training_agents, noise_agents, bid):
		'''
		Fills all orders found possible, notifies agents and updates the limit order book
		Args:
			px: numpy array, prices at which to orders exist in order book
			vol: numpy array, corresponding volumes for which orders exist in order book
			idx: int, maximum index in array that prices can be cleared
			training_agents: dict, key is agentId, value is the agent object
			noise_agents: dict, key is agentId, value is the agent object
			bid: boolean, true if filling buy orders
		'''
		if bid:
			orders = self.buy_actions
			action = 0
		else:
			orders = self.sell_actions
			action = 1

		orders_left_to_fill = self.num_trades_filled
		for i in range(idx):
			temp_px = px[i]
			acts = orders[temp_px]

			temp_vol = int(vol[i])
			if temp_vol <= orders_left_to_fill:
				orders_left_to_fill -= temp_vol
				del orders[temp_px]
				for a in acts:
					agId, oId, _ = a
					if agId in training_agents:
						training_agents[agId].filled_order( (oId, action, self.price) )
					else:
						noise_agents[agId].filled_order( (oId, action, self.price) )
			else:
				fill_orders = np.random.choice(temp_vol, orders_left_to_fill, replace=False)
				for a in [acts[j] for j in fill_orders]:
					agId, oId, _ = a
					if agId in training_agents:
						training_agents[agId].filled_order( (oId, action, self.price) )
					else:
						noise_agents[agId].filled_order( (oId, action, self.price) )
				orders[temp_px] = [acts[j] for j in (set(list(range(temp_vol))) - set(fill_orders))]

	def add_to_book(self, agent_Id, agent_action):
		'''
		takes agent IDs and actions and inserts in limit order book
		Args:
			agent_Id: string, which agent sent order
			agent_action: tuple, specifics of order with orderId, action (buy/sell), price, and duration
		'''
		oId, action, price, duration = agent_action
		if action == 0: # buy order
			if price in self.buy_actions: # price level already exists, add to waiting orders
				self.buy_actions[price].append( (agent_Id, oId, duration) )
			else:
				self.buy_actions[price] = [(agent_Id, oId, duration)]
		elif action == 1: # sell order
			if price in self.sell_actions: # price level already exists, add to waiting orders
				self.sell_actions[price].append( (agent_Id, oId, duration) )
			else:														# new price is current lowest unfilled ask
				self.sell_actions[price] = [(agent_Id, oId, duration)]
		

	def cancel_orders(self, n):
		'''
		cancels all orders set to expire on next round before start of next round
		Args:
			n: int, round number of game
		'''

		# cancel actions in buy actions
		for k, v in self.buy_actions.items():
			self.buy_actions[k] = [tup for tup in v if tup[2] > n]
		self.buy_actions = {k:v for k,v in self.buy_actions.items() if len(v) > 0}

		# cancel actions in sell actions
		for k, v in self.sell_actions.items():
			self.sell_actions[k] = [tup for tup in v if tup[2] > n]
		self.sell_actions = {k:v for k,v in self.sell_actions.items() if len(v) > 0}

		remaining_bid = self.buy_actions.keys()
		if len(remaining_bid) > 0:
			self.max_bid = max(remaining_bid)
		else:
			self.max_bid = -np.inf
		remaining_ask = self.sell_actions.keys()
		if len(remaining_ask) > 0:
			self.min_ask = min(remaining_ask)
		else:
			self.min_ask = np.inf

	def create_market(self):
		'''
		Turns dictionaries of buy/sell actions into a 4xk numpy array as market state
		'''
		bid_px_vol = self.price_volume(bid=True)
		ask_px_vol = self.price_volume(bid=False)

		# create new market state
		self.current_market = self.order_book_to_market(bid_px_vol, ask_px_vol)

	def price_volume(self, bid=True):
		'''
		Function that returns price, volume pairs for bid/ask in numpy array
		Args:
			d: dictionary, contains price, volume pairs with price as key
			bid: bool, is the dict being passed of bid price/volume
		Returns:
			2 lists, one with all prices, one with all volumes, same index are associated
		'''
		if bid:
			orders = self.buy_actions
		else:
			orders = self.sell_actions

		price_volume = []
		for k, v in orders.items():
			price_volume.append( (k, len(v)) )

		price_volume.sort(key=lambda x: x[0], reverse=bid)

		return np.array(price_volume)

	def order_book_to_market(self, bid_px_vol, ask_px_vol):
		'''
		Turns current limit order book into a state of market of 4 n-vectors
		Args:
			bid_px_vol: list, list of tuples of current bid price/volumes in descending order
			ask_px_vol: list, list of tuples of current ask price/volumes in descending order
			n: int, how deep into orders on each side to go to create market state (0 pad if less than n)
		Returns:
			market state as a 4xk numpy array
		'''
		# bid_px_vol_array = np.array(bid_px_vol)
		nrow = bid_px_vol.shape[0]
		if len(bid_px_vol) > 0: # non-empty buy orders in book
			# return top k orders with zero-padding if number of orders < k
			bid_px_vol = np.pad(bid_px_vol, ((0, max(0, self.k-nrow)), (0,0)), 'constant')
		else:
			bid_px_vol = np.zeros(shape=(self.k,2))

		# ask_px_vol_array = np.array(ask_px_vol)
		nrow = ask_px_vol.shape[0]
		if len(ask_px_vol) > 0: # non-empty ask orders in book
			# return top k orders with zero-padding if number of orders < k
			ask_px_vol = np.pad(ask_px_vol, ((0, max(0, self.k-nrow)), (0,0)), 'constant')
		else:
			ask_px_vol = np.zeros(shape=(self.k,2))

		return np.hstack( (bid_px_vol[:self.k, :], ask_px_vol[:self.k,:]) ).T




















