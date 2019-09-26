class BaseAgent():
	current_val = None # value of current holdings?
	informed = False
	next_dividend = None
	dividend_amt = None

	def __init__(self):
		self.current_val = 0

	# to be implemented in each class on how to act given current state
	def act(self, obs=None):
		raise NotImplementedError()

	# update whether agent knows about next dividend
	def update_info(self, next_dividend, dividend_amt):
		self.next_dividend = next_dividend
		self.dividend_amt = dividend_amt

	# update current wealth
	def update_val(self, amt):
		self.current_val += amt

	# def test_update(self):
	# 	self.current_val += 1

	# helper function to convert observation to vector
	def state_parser(self, obs):
		'''
		Args:
			obs is a tuple of dictionaries containing 
				'bid': bid price,
				'bid depth': total number of buy orders
				'ask': ask price
				'ask depth': total number of sell orders
				'round': n
		'''
		state_vector = []
		for o in obs:
			state_vector.append(o['bid'])
			state_vector.append(o['bid depth'])
			state_vector.append(o['ask'])
			state_vector.append(o['ask depth'])

		state_vector.append(obs[len(obs)-1]['round'])
		return state_vector