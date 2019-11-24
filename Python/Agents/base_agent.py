class BaseAgent():
	agentId = None 		# signature of agent
	cash = None 		# amount of cash agent has
	assets = None 		# number of shares of asset agent has
	open_orders = None 	# dict, key is tradeID, value is action, price, duration tuple

	def __init__(self, ID, agent_type):
		'''
		Initialize agent
		Args:
			ID: int, unique identifier for agent
			agent_type: string, 'noise', 'informed', or 'uninformed' to know what kind of agent
		'''
		self.cash = 0
		self.assets = 0
		if agent_type == 'noise':
			self.agentId = 'n' + str(ID)
		elif agent_type == 'informed':
			self.agentId = 'i' + str(ID)
		elif agent_type == 'uninformed':
			self.agentId = 'u' + str(ID)
		self.open_orders = {}

	def act(self):
		''' To be implemented in each class on how to act given current state '''
		raise NotImplementedError()

	def update_val(self, dividend_amt):
		'''
		Update agent's current wealth by dividend amount
		Args:
			dividend_amt: int, amount to increase cash holdings by per unit of asset held
		'''
		self.cash += max(self.assets, 0) * dividend_amt

	def cancel_orders(self, n):
		'''
		Updates list of open orders based on time before acting at each time step
		Args:
			n: int, current time step
		''' 
		self.open_orders = {k:v for k,v in self.open_orders.items() if v[2] > n-1}


	def filled_order(self, order):
		'''
		Updates list of open orders based on filled orders when called
		Args:
			order: tuple of tradeId, action taken, price it was filled at 
		'''
		Id, action, price = order
		if action == 0: # buy order
			self.cash -= price
			self.assets += 1
		elif action == 1: # sell order
			self.cash += price
			self.assets -= 1

		del self.open_orders[Id] # order filled, remove from open orders


	# def test_update(self):
	# 	self.current_val += 1

	# helper function to convert observation to vector
	def state_parser(self, obs=None, n=0):
		'''
		Args:
			obs: , representation of current state passed to informed agents
		'''
		raise NotImplementedError()