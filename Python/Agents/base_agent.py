class BaseAgent():
	agentId = None 		# signature of agent
	cash = None 		# amount of cash agent has
	assets = None 		# number of shares of asset agent has
	informed = False 	# may not need
	next_dividend = -1 	# -1 if no info known - can't act? ideally would be None
	dividend_amt = 0 	# 0 for no info known - ideally is None
	open_orders = None 	# dict, key is tradeID, value is action, price, duration tuple

	def __init__(self):
		self.cash = 0
		self.assets = 0
		self.open_orders = {}

	def act(self, obs=None):
		''' To be implemented in each class on how to act given current state '''
		raise NotImplementedError()

	# should be in informed agent
	def update_info(self, next_dividend, dividend_amt):
		'''
		Update what agent knows about next dividend
		Args:
			next_dividend: int, time of next dividend
			dividend_amt: int, amount of next dividend (can be + or -)
		'''
		self.next_dividend = next_dividend
		self.dividend_amt = dividend_amt

	def update_val(self, dividend_amt):
		'''
		Update agent's current wealth by dividend amount
		Args:
			dividend_amt: int, amount to increase cash holdings by per unit of asset held
		'''
		self.cash += self.assets * dividend_amt

	def cancelled_order(self, n):
		'''
		Updates list of open orders based on time before acting at each time step
		Args:
			n: int, current time step
		''' 
		self.open_orders = {k:v for k,v in self.open_orders.items() if v[2] > n}


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
	def state_parser(self, obs):
		'''
		Args:

		'''