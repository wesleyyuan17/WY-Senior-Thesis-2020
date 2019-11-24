import numpy as np
from .base_agent import BaseAgent

class SimpleAgent(BaseAgent):
	'''
	Simple agent that takes in current market state and uses it (plus agent-specific knowledge if any)
	to decide whether to buy/sell/nothing as market prices

	Informed - should trade in direction of dividend amount up to some threshold?
	'''
	informed = None		# is this trader informed or not?
	prev_value = None 	# previous value of portfolio
	k = None			# depth of market space being passed, used by state parser
	memory = None		# memory of all states this agent previously experienced, for batch learning
	main_DQN = None		# DQN that learns actions
	target_DQN = None	# DQN that acts as target in double q learning

	def __init__(self, ID, state_size, informed):
		if informed:
			super().__init__(ID, 'informed')
			self.next_dividend = -1 # time of next dividend
			self.dividend_amt = 0 # amount of next dividend
			# self.trade_until = -1 # time-step until which to keep trading as specified
			# self.trade_param = 1 # parameter of geometric distribution for how long to continue sending market orders
		else:
			super().__init__(ID, 'uninformed')

		self.informed = informed
		self.prev_value	= 0
		self.k = state_size

	def act(self, obs, price, last_executed, n):
		'''
		Takes an based on current market state
		Args:
			obs: 4xk array, represents bid price/volume in decreasing order and ask price/volume in increasing order
			price: double, market-clearing price from last step
			last_executed: int, number of orders filled at market-clearing price at last step
			n: int, the time step
		'''
		current_state = self.state_parser(obs, price, last_executed)
		# do stuff with the state now - pass through neural net?

	def state_parser(self, obs, price, last_executed):
		'''
		Creates current state from market observation and instance variables
		Args:
			obs: 4xk array, represents bid price/volume in decreasing order and ask price/volume in increasing order
			n: int, the time step
		'''

		# creates addition to market state with agent-specific info and zero-pads to match length
		addition = np.array([[price, self.cash, self.next_dividend], 
							 [last_executed, self.assets, self.dividend_amt]])
		addition = np.pad(addition, ((0, 0), (0,self.k-3)), 'constant')

		# concatenate along horizontal axis
		return np.vstack( (obs, addition) )

	def update_info(self, next_dividend, dividend_amt):
		'''
		Update what this agent knows about next dividend
		Args:
			next_dividend: int, time of next dividend
			dividend_amt: int, amount of next dividend (can be + or -)
		'''
		self.next_dividend = next_dividend
		self.dividend_amt = dividend_amt

