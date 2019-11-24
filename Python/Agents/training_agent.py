import numpy.random as npr
from .base_agent import BaseAgent
# from ..HelperClasses import Memory#, DQN

class TrainingAgent(BaseAgent):
	# informed = False 	# may not need
	next_dividend = -1 	# -1 if no info known - can't act? ideally would be None
	dividend_amt = 0 	# 0 for no info known - ideally is None
	prev_value = None 	# previous value of portfolio
	memory = None		# memory of all states this agent previously experienced
	main_DQN = None		# DQN that learns actions
	target_DQN = None	# DQN that acts as target in double q learning

	def __init__(self, ID, agent_type):
		super().__init__(ID, agent_type)
		self.next_dividend = -1
		self.dividend_amt = 0
		self.prev_value = 0
		# self.memory = Memory()
		# self.main_DQN = DQN()
		# self.target_DQN = DQN()

	def act(self, current_market, n):
		_

	def update_info(self, next_dividend, dividend_amt):
		'''
		Update what agent knows about next dividend
		Args:
			next_dividend: int, time of next dividend
			dividend_amt: int, amount of next dividend (can be + or -)
		'''
		self.next_dividend = next_dividend
		self.dividend_amt = dividend_amt

	def state_parser(self, current_market, n):
		_