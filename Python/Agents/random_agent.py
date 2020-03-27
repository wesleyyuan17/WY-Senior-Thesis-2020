import numpy.random as npr
from .base_agent import BaseAgent

class RandomAgent(BaseAgent):
	'''
	'''
	def __init__(self, n_actions, ID):
		'''
		Initialize agent to act within a set of n_action actions with own ID
		'''
		super().__init__(ID, 'noise')
		self.mean_duration = 10
		self.n_actions = n_actions

	# determine action based on current state
	def act(self, obs, n): # how obs is represented may change
		'''
		Takes a random action based on current market state
		Args:
			obs: 4xk array, represents bid price/volume in decreasing order and ask price/volume in increasing order
			n: int, the time step
		'''
		# update open orders for next round
		super()._cancel_orders(n)

		# take action
		oId = self.agentId + '_' + str(n) # set order ID
		action = npr.randint(2)
		if action == 0: # buy order

			price = self.pick_price(obs[2,0], n, True) #  + npr.randint(-7, 3) # number between -7 to 2 inclusive
			duration = n + npr.geometric(p=1/self.mean_duration)
			
			self.open_orders[oId] = (action, price, duration) # store current order
		elif action == 1: # sell order

			price = self.pick_price(obs[0,0], n, False) # + npr.randint(-2, 8) # number between -2 to 7 inclusive
			duration = n + npr.geometric(p=1/self.mean_duration)

			self.open_orders[oId] = (action, price, duration) # store current order
		else: # do nothing
			price = 0
			duration = 0
		
		return (oId, action, price, duration)

	def pick_price(self, ref_price, n, bid):
		if bid:
			phi = 1
		else:
			phi = -1

		if npr.rand() < 0.1 and n > 10: # cross bid/ask spread 10% of the time or first 10 to ensure filled LOB
			return ref_price + phi*10 # 10 to be sure it crosses?
		else:
			return ref_price - phi*npr.randint(0,11) # up to 10 to have a deep enough book?
			# phi*10*npr.rand() for continuous
