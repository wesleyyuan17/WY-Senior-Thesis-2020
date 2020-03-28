import numpy as np
import torch

class Memory():
	max_mem_size = None 	# maximum size of memory
	market_history = None 	# memory of past states - Change to reset after each dividend payout?
	batch_size = None 		# number of states to return per batch call
	window_size = None 		# number of frames that make up a state
	count = None			# number of frames currently in memory
	actions = None 			# previous actions corresponding to states in market_history
	rewards = None			# previous rewards for actions taken

	# Initializes values of memory object
	def __init__(self, mem_size=1000000, frame_height=6, frame_width=10, 
				 batch_size=32, window_size=1):
		'''
		Args:
			mem_size: set maximum size of memory
			batch_size: number of states to return per batch call
		'''
		self.max_mem_size = mem_size
		self.batch_size = batch_size
		self.window_size = window_size
		self.frame_height = frame_height
		self.frame_width = frame_width
		self.count = 0
		self.current = 0
		# self.moving_average = 0 # 20-period rolling window average of mid-price
		# self.moving_stddev = 1 # 20-period rolling window std dev of mid-price
		# self.price_window = [] # for calculating moving averages

		# pre-allocate memory for memories
		self.market_history = torch.empty((self.max_mem_size, self.frame_height, self.frame_width), dtype=torch.float32)
		self.actions = torch.empty(self.max_mem_size, dtype=torch.int32)
		self.rewards = torch.empty(self.max_mem_size, dtype=torch.float32)

		# pre-allocate memory for minibatch objects
		self.states = torch.empty((self.batch_size, self.window_size, self.frame_height, self.frame_width), dtype=torch.float32)
		self.new_states = torch.empty((self.batch_size, self.window_size, self.frame_height, self.frame_width), dtype=torch.float32)
		self.indices = np.empty(self.batch_size, dtype=np.int32)

	def add_memory(self, action, market_state, reward): # add reward
		''' 
		Add new experience to memory
		Args:
			market_state: numpy array, representing new state to be added to memory
		'''
		# mid = (market_state[0][0] + market_state[2][2]) / 2
		self.market_history[self.current, ...] = torch.tensor(market_state) # - mid # self.moving_average) / self.moving_stddev # normalized
		self.actions[self.current] = action
		self.rewards[self.current] = reward
		self.count = min(self.count + 1, self.max_mem_size)
		self.current = (self.current + 1) % self.max_mem_size

		# updates on rolling window price history
		# mid = (market_state[0][0] + market_state[2][0]) / 2
		# self.price_window.append(mid)
		# if len(self.price_window) > 20: # 100 to minimize probability of stddev = 0?
		# 	self.price_window.pop(0)
		# self.moving_average = np.mean(self.price_window)
		# self.moving_stddev = np.std(self.price_window)
		# if self.moving_stddev == 0:
		# 	self.moving_stddev = 1000

	# get sequence of memory as current state
	def get_current_state(self):
		return self._get_state(self.current-1)[None, ...]

	def _get_state(self, index):
		'''
		Return state for specified index whose last frame is at index+1
		Args:
			index: int, index of last frame of state requested
		'''
		if self.count is 0:
			raise ValueError("The replay memory is empty!")
		if index < self.window_size - 1:
			raise ValueError("Index must be min 3")
		return self.market_history[index-self.window_size+1:index+1, ...]

	# find indices that have valid contiguous frames at random points
	def _get_valid_rand_indices(self):
		for i in range(self.batch_size):
			while True:
				index = np.random.randint(self.window_size, self.count - 1)
				if index < self.window_size:
					continue
				if index >= self.current and index - self.window_size <= self.current:
					continue
				break
			self.indices[i] = index

	# return indices of last batch_size contiguous frames
	def _get_valid_seq_indices(self):
		if self.current >= self.batch_size + self.window_size:
			self.indices = np.arange(self.current-self.window_size, self.current)
		elif self.count >= self.max_mem_size:
			# wrap around
			temp_ok = self.current - self.window_size
			self.indices[:temp_ok] = np.arange(self.window_size, self.current)
			self.indices[temp_ok:] = np.arange(self.count-(self.batch_size - temp_ok)-1, self.count-1)
		else:
			i = 0
			available = self.current - self.window_size
			while (i+1)*available < self.batch_size:
				self.indices[i*available:(i+1)*available] = np.arange(self.window_size, self.current)
			addition = self.batch_size - i*available
			self.indices[i*available:] = np.arange(self.window_size, self.window_size+addition)

	def get_minibatch(self, batch_type='rand'):
		'''
		Returns slices of memory to return for training
		Args:
			batch_type: str, if seq - gets last batch_size indices, if rand - gets random indices
		Returns:
			3 numpy arrays of state, action, reward, new state
		'''
		if self.count < self.window_size:
			raise ValueError('Not enough memories to get minibatch')

		if batch_type == 'seq':
			self._get_valid_seq_indices()
		elif batch_type == 'rand':
			self._get_valid_rand_indices()
		else:
			raise ValueError('Unrecognized batch type')

		for i, idx in enumerate(self.indices):
			self.states[i] = self._get_state(idx-1)
			self.new_states[i] = self._get_state(idx)

		return [self.states, 
				self.actions[self.indices+1], 
				self.rewards[self.indices+1],
				self.new_states]

# assume window is 20 - plan dilation around this window size
# act method should assume always exploit
class DQN(torch.nn.Module): 
	def __init__(self, n_actions, n_filters=[32, 64, 512], window_size=1): # maybe add more?
		'''
		Args:
			n_actions: int, number of valid actions for a state i.e. output dim
			n_filters: list, number of filters in each convolutional layer
			window_size: number of frames that make up state i.e. input dim for first
						 convolutional layer *subject to change* - starting with 4 for simplicity
		'''
		super().__init__()
		
		self.action_space = n_actions

		'''
		1st layer: 4 input channels, 32 filters of 2x2 w/ stride 2x1, apply rectifier
		Output: 32x3x9

		2nd layer: 32 input channels, 64 filters of 3x3 w/ stride 1, apply rectifier
		Output: 64x1x7

		3rd layer: 64 input channels, 1024 filters of 1x7 w/ stride 1, apply rectifier
		Output: 1024x1x1
		'''

		# self.conv = torch.nn.Sequential(
		# 	# 1st layer
		# 	torch.nn.Conv2d(window_size, n_filters[0], kernel_size=(2,2), stride=(2,1), bias=True),
		# 	torch.nn.ReLU(),
		# 	# 2nd layer
		# 	torch.nn.Conv2d(n_filters[0], n_filters[1], kernel_size=(3,3), stride=1, bias=True),
		# 	torch.nn.ReLU(),
		# 	# 3rd layer
		# 	torch.nn.Conv2d(n_filters[1], n_filters[2], kernel_size=(1,7), stride=1, bias=True),
		# 	torch.nn.ReLU(),
		# 	)

		# simplified
		self.conv = torch.nn.Sequential(
			# 1st layer
			torch.nn.Conv2d(window_size, n_filters[0], kernel_size=(2,2), stride=1, bias=True),
			torch.nn.ReLU(),
			# torch.nn.Tanhshrink(),
			# 2nd layer
			# torch.nn.Conv2d(n_filters[0], n_filters[1], kernel_size=(1,3), stride=1, bias=True),
			# torch.nn.ReLU()
			)

		self.linear = torch.nn.Sequential(
			# 2nd layer
			torch.nn.Linear(n_filters[0], n_filters[1]),
			torch.nn.ReLU(),
			# torch.nn.Tanhshrink(),
			torch.nn.Linear(n_filters[1], n_filters[1]),
			torch.nn.ReLU(),
			)

		# advantage stream
		self.advantage = torch.nn.Sequential(
			torch.nn.Linear(n_filters[1], n_filters[1]),
			torch.nn.ReLU(),
			# torch.nn.Tanhshrink(), # like a smooth, symmetric ReLU? - negative Q-values unlike with RelU
			torch.nn.Linear(n_filters[1], self.action_space)
			)

		# value stream
		self.value = torch.nn.Sequential(
			torch.nn.Linear(n_filters[1], n_filters[1]),
			torch.nn.ReLU(),
			# torch.nn.Tanhshrink(), # like a smooth, symmetric ReLU? - negative Q-values unlike with RelU
			torch.nn.Linear(n_filters[1], 1)
			)

	def forward(self, x):
		'''
		Passes x through the DQN, returns the resulting Q-value output
		Args:
			x: A ([dimensions]) tensor representing current state
		Returns:
			Vector of predicted value of taking any of n_action actions given current state
		'''
		# x = torch.tensor(x).float()
		x = self.conv(x) # pass through convolutional layer
		x = self.linear(torch.squeeze(x))
		# print(x.shape)
		value, advantage = torch.squeeze(x.clone()), torch.squeeze(x.clone()) # torch.chunk(torch.squeeze(x), 2)
		value = self.value(value)
		advantage = self.advantage(advantage)
		# print(value.shape, advantage.shape)
		return value + advantage - advantage.mean()
		# return torch.squeeze(x)

	def act(self, state):
		'''
		Takes a state and decides how to act for next step assuming always exploiting
		Args:
			state: A ([dimensions]) tensor corresponding to the current state representation
		Returns:
			action: integer, index corresponding to action with greatest predicted q-value
		'''
		q_val = self.forward(state)
		print("Estimated Q-values:", q_val.tolist())
		action = torch.argmax(q_val)

		return action























