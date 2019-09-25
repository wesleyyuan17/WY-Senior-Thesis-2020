class BaseAgent():
	current_val = None # value of current holdings?
	informed = False
	next_dividend = None
	dividend_amt = None

	def __init__(self):
		self.current_val = 0

	def act(self, obs):
		raise NotImplementedError()

	def update_info(self, next_dividend, dividend_amt):
		self.next_dividend = next_dividend
		self.dividend_amt = dividend_amt

	def update_val(self, amt):
		self.current_val += amt
	# def test_update(self):
	# 	self.current_val += 1