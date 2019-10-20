import numpy as np

class Memory():
	max_mem_size = None 	# maximum size of memory
	market_history = None 	# memory of past states
	batch_size = None 		# number of states to return per batch call
	window = None 			# number of frames that make up a state
	count = None			# number of frames currently in memory
	state = None			# current state as window-sized stack of previous market states

	# Initializes values of memory object
	def __init__(self, mem_size=100000, batch_size=20, window=4):
		'''
		Args:
			mem_size: set maximum size of memory
			batch_size: number of states to return per batch call
		'''
		self.max_mem_size = mem_size
		self.market_history = []
		self.batch_size = batch_size
		self.window = window
		self.count = 0
		self.state = np.zeros(shape=(4,10,self.window))

	# Add new experience to memory
	def add_memory(self, market_state):
		''' 
		Args:
			market_state: dict representing new state to be added to memory
		'''
		self.market_history.append(market_state)
		if self.count >= self.max_mem_size:
			self.market_history.pop(0)
		self.count = min(self.count + 1, self.max_mem_size)
		self.state = np.append(self.state[:, :, 1:], market_state[:, :, None], axis=2)

	# get sequence of memory as current state
	def get_current_state(self):
		return self.market_history[self.count - self.window:, ...]

	# randomly select slices of memory to return for training
	def get_rand_minibatch(self):
		idx = np.random.choice(self.count - self.window, batch_size, replace=False)
		batch = []
		for i in idx:
			batch.append( (self.market_history[i:i+self.window, ...]) )
		return batch

	# get last batch_size number of states from memory
	def get_seq_minibatch(self):
		batch = []
		for i in range(batch_size):
			idx = self.count - i - self.window
			batch.append( (self.market_history[idx:idx+self.window, ...]) )
		return batch

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

	def __init__(self):
		self.buy_actions = {}
		self.sell_actions = {}
		self.min_ask = np.inf
		self.max_bid = -np.inf
		self.current_market = np.zeros(shape=(4,10))

	def update(self, agent_actions, training_agents, noise_agents, n):
		'''
		called in game to perform necessary updates to market and agents
		Args:
			actions: list, contains agentId with associated action for this round for each agent
			training_agents: dict, key is agentId, action is 
		'''
		self.get_actions(agent_actions, training_agents, noise_agents)

		self.cancel_orders(n)
		self.create_market()

		return self.current_market

	def get_actions(self, agent_actions, training_agents, noise_agents):
		# get actions from agents
		for agId, act in agent_actions.items():
			oId, action, price, duration = act
			if action == 0:															# action is buy order

				if price > self.min_ask: 											# buy order fills current ask
					oAgent, otId, _ = self.sell_actions[self.min_ask].pop() 		# get filled order

					if len(self.sell_actions[self.min_ask]) == 0: 					# no more sell orders at this price
						del self.sell_actions[self.min_ask]
						remaining_ask = self.sell_actions.keys()					# update min ask price b/c previous was filled
						if len(remaining_ask) > 0:
							self.min_ask = min(remaining_ask)
						else:
							self.min_ask = np.inf

					if agId in training_agents:										# notify current agent of filled order
						training_agents[agId].filled_order( (oId, 1, self.min_ask) )
					else:
						noise_agents[agId].filled_order( (oId, 1, self.min_ask) )
					if oAgent in training_agents:									# notify other agent of filled order
						training_agents[oAgent].filled_order( (otId, 1, self.min_ask) )
					else:
						noise_agents[oAgent].filled_order( (otId, 1, self.min_ask) )

				elif price in self.buy_actions:										# buy order price level exists
					self.buy_actions[price].append( (agId, oId, duration) )
				else:																# buy order price level does not exist, create
					self.buy_actions[price] = [(agId, oId, duration)]
					if price > self.max_bid:										# new price is current highest unfilled bid
						self.max_bid = int(price)

			elif action == 1:														# action is sell order
			
				if price < self.max_bid:											# sell order fills current bid
					oAgent, otId, _ = self.buy_actions[self.max_bid].pop()			# get filled order
					if len(self.buy_actions[self.max_bid]) == 0:					# no  more buy orders at this price
						del self.buy_actions[self.max_bid]
						remaining_bid = self.buy_actions.keys()						# update max bid price b/c previous was filled
						if len(remaining_bid) > 0:
							self.max_bid = max(remaining_bid)
						else:
							self.max_bid = -np.inf

					if agId in training_agents:										# notify current agent of filled order
						training_agents[agId].filled_order( (oId, 1, self.min_ask) )
					else:
						noise_agents[agId].filled_order( (oId, 1, self.min_ask) )
					if oAgent in training_agents:									# notify other agent of filled order
						training_agents[oAgent].filled_order( (otId, 0, self.max_bid) )
					else:
						noise_agents[oAgent].filled_order( (otId, 0, self.max_bid) )

				elif price in self.sell_actions:									# sell order price level exists
					self.sell_actions[price].append( (agId, oId, duration) )
				else:																# sell order price level does not exist, create
					self.sell_actions[price] = [(agId, oId, duration)]
					if price < self.min_ask:										# new price is current lowest unfilled ask
						self.min_ask = int(price)

	def cancel_orders(self, n):
		for k, v in self.buy_actions.items():
			self.buy_actions[k] = [tup for tup in v if tup[2] > n+1]
		self.buy_actions = {k:v for k,v in self.buy_actions.items() if len(v) > 0}

		for k, v in self.sell_actions.items():
			self.sell_actions[k] = [tup for tup in v if tup[2] > n+1]
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
		bid_px_vol = self.dict_to_list(self.buy_actions, bid=True)
		ask_px_vol = self.dict_to_list(self.sell_actions, bid=False)

		# create new market state
		self.current_market = self.order_book_to_market(bid_px_vol, ask_px_vol, 10)

	def dict_to_list(self, d, bid=True):
		'''
		Function that converts a dictionary of price, volume pairs and returns each in its own list
		Args:
			d: dictionary, contains price, volume pairs with price as key
			bid: bool, is the dict being passed of bid price/volume
		Returns:
			2 lists, one with all prices, one with all volumes, same index are associated
		'''
		price_volume = []
		for k, v in d.items():
			price_volume.append( (k, len(v)) )

		price_volume.sort(key=lambda x: x[0], reverse=bid)

		return price_volume

	def order_book_to_market(self, bid_px_vol, ask_px_vol, k):
		'''
		Turns current limit order book into a state of market of 4 n-vectors
		Args:
			bid_px_vol: list, list of tuples of current bid price/volumes in descending order
			ask_px_vol: list, list of tuples of current ask price/volumes in descending order
			n: int, how deep into orders on each side to go to create market state (0 pad if less than n)
		Returns:
			market state as a 4xn numpy array
		'''
		bid_px_vol_array = np.array(bid_px_vol)
		nrow = bid_px_vol_array.shape[0]
		if len(bid_px_vol_array) > 0:
			bid_px_vol_array = np.pad(bid_px_vol_array, ((0, max(0, 10-nrow)), (0,0)), 'constant')
		else:
			bid_px_vol_array = np.zeros(shape=(10,2))

		ask_px_vol_array = np.array(ask_px_vol)
		nrow = ask_px_vol_array.shape[0]
		if len(ask_px_vol_array) > 0:
			ask_px_vol_array = np.pad(ask_px_vol_array, ((0, max(0, 10-nrow)), (0,0)), 'constant')
		else:
			ask_px_vol_array = np.zeros(shape=(10,2))

		return np.hstack( (bid_px_vol_array[:k, :], ask_px_vol_array[:k,:]) ).T




















