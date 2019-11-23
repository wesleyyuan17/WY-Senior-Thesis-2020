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

class DQN():
	_