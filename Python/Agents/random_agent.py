import numpy.random as npr
from .base_agent import BaseAgent

class RandomAgent(BaseAgent):
	'''
	'''
	def __init__(self, n_actions, ID, *args, **kwargs):
		'''
		'''
		super().__init__(*args, **kwargs)
		self.agentId = 'n' + str(ID)
		self.mean_duration = 10
		self.n_actions = n_actions

	# determine action based on current state
	def act(self, obs, n): # how obs is represented may change
		'''
		'''
		# update open orders
		super().cancelled_order(n)

		# take action
		oId = self.agentId + '_' + str(n) # set order ID
		action = npr.randint(3)
		if action == 0: # buy order

			price = obs[2,0] + npr.randint(-7, 3) # number between -7 to 2 inclusive
			duration = n + npr.geometric(p=1/self.mean_duration)
			
			self.open_orders[oId] = (action, price, duration) # store current order

		elif action == 1: # sell order

			price = obs[0,0] + npr.randint(-2, 8) # number between -2 to 7 inclusive
			duration = n + npr.geometric(p=1/self.mean_duration)

			self.open_orders[oId] = (action, price, duration) # store current order

		else: # do nothing
			price = 0
			duration = 0
		
		return (oId, action, price, duration)