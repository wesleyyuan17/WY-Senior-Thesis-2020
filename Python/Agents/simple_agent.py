import numpy as np
import torch

from .base_agent import BaseAgent
from .HelperClasses import *

class SimpleAgent(BaseAgent):
	'''
	Simple agent that takes in current market state and uses it (plus agent-specific knowledge if any)
	to decide whether to buy/sell/nothing as market prices

	Informed - should trade in direction of dividend amount up to some threshold?
	'''
	informed = None				# is this trader informed or not?
	prev_value = None 			# previous value of portfolio
	dividend_paid = None		# boolean - true if agent just received dividend
	k = None					# depth of market space being passed, used by state parser
	memory = None				# memory of all states this agent previously experienced, for batch learning
	last_action = 2				# last action taken stored to add to memory
	main_DQN = None				# DQN that learns actions
	target_DQN = None			# DQN that acts as target in double q learning
	opt = None					# optimizer for running gradient descent
	_FINAL_EPS_ = 0.1			# long-term floor probability of taking a random action
	_MEM_START_SIZE_ = 500	 	# number of steps of random action
	_TN_UPDATE_FREQ_ = 1000 	# Every C steps, update target network
	_BATCH_SIZE_ = 32			# batch size for batch training - equal to expected duration between dividends
	_GAMMA_ = 0.99 				# discount factor for Bellman equation
	_ALPHA_ = 0.000001 			# learning rate for parameter updates

	def __init__(self, ID, state_size, informed, eval):
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
		self.memory = Memory(batch_size=self._BATCH_SIZE_)
		if eval:
			# evaluation mode - no learning
			self.main_DQN = torch.load(self.agentId+'NN.pth')
			self.main_DQN.eval()
		else:
			# training mode
			self.main_DQN = DQN(n_actions=30) # action space of 30 buy/sell/nothing by duration 1 through 10
			self.target_DQN = DQN(n_actions=30)
			self.opt = torch.optim.Adam(self.main_DQN.parameters(), lr=self._ALPHA_)
			self.loss = 0

	def update_info(self, next_dividend, dividend_amt):
		'''
		Update what this agent knows about next dividend, indicates in memory object that new dividend is out
		Args:
			next_dividend: int, time of next dividend
			dividend_amt: int, amount of next dividend (can be + or -)
		'''
		self.next_dividend = next_dividend
		self.dividend_amt = dividend_amt
		self.dividend_paid = True

	def current_value(self):
		'''
		Returns current value of portfolio of agent
		'''
		return self.prev_value

	def training_loss(self):
		'''
		Returns training loss over life of agent
		'''
		return self.loss

	def save_DQN(self):
		torch.save(self.main_DQN, self.agentId+'NN.pth')

	def act(self, obs, price, last_executed, n):
		'''
		Updates self memory/reward and takes an action based on current market state
		Args:
			obs: 4xk array, represents bid price/volume in decreasing order and ask price/volume in increasing order
			price: double, market-clearing price from last step
			last_executed: int, number of orders filled at market-clearing price at last step
			n: int, the time step
		'''
		# update open orders for next round
		super()._cancel_orders(n)

		current_state = self._state_parser(obs, price, last_executed)

		# update memory with new transition based on new info
		self._update_memory(current_state, price)

		# take random action according to exploration/exploitation
		if np.random.random() < self._eps_scheduler(n):
			raw_action = np.random.randint(30)
		else:
			raw_action = self.main_DQN.act(self.memory.get_current_state()) # act implemented to that it assumes no exploration

			# make updates if in training mode
			if not eval:
				self.loss = self._learn()
				# loss_list.append(loss)

				# update target network as needed
				if n % self._TN_UPDATE_FREQ_ == 0:
					self._update_target_dqn()

		# map raw action to corresponding action/duration
		self.last_action = raw_action
		action = int(raw_action / 10) # 0-9: 0 (buy), 10-19: 1 (sell), 20-29: 2 (no action)
		duration = raw_action % 10 + 1 # duration in [1,10]

		if action == 0: # market buy order
			price = obs[2,0] + 10 # submit order at inside-ask + 10
		elif action == 1: # market sell order
			price = obs[0,0] - 10 # submit order at inside-bid - 10

		oId = self.agentId + '_' + str(n) # set order ID
		self.open_orders[oId] = (action, price, duration) # add to current open orders

		# for debugging
		print("next dividend amount:", self.dividend_amt, "action taken:", action)

		return (oId, action, price, duration)

	def _state_parser(self, obs, price, last_executed):
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

	def _eps_scheduler(self, n):
		'''
		Determines epsilon at given time step
		Args:
			n: int, time period
		'''
		return max(self._MEM_START_SIZE_ / n, self._FINAL_EPS_)

	def _clip_reward(self, reward):
		'''
		Reduces reward to sign for more stable learning
		Args:
			reward: double, value to be clipped
		'''
		if reward > 0:
			return 1
		elif reward == 0:
			return 0
		else:
			return -1

	def _update_memory(self, market_state, price):
		'''
		Adds previous action and resulting reward to memory, updates own state accordingly
		Args:
			obs: 4xk array, represents bid price/volume in decreasing order and ask price/volume in increasing order
			price: double, market-clearing price from last step
		'''
		new_value = self.assets*price + self.cash
		reward = self._clip_reward(new_value - self.prev_value) # sign of change in cumulative net P&L
		self.memory.add_memory(self.last_action, market_state, reward) #, self.dividend_paid) how would taking dividend_paid work? Also need to add other things later?

		self.dividend_paid = False
		self.prev_value = new_value

	def _update_target_dqn(self):
		'''
		Updates parameters of target DQN
		'''
		torch.save(self.main_DQN, self.agentId+'NN.pth')
		self.target_DQN = torch.load(self.agentId+'NN.pth')
		self.target_DQN.eval()

	def _learn(self):
		'''
		Implements action taking, loss calculation, parameter updating
		Returns:
			loss: double for loss value
		'''
		# Draw a minibatch from the replay memory
		states, actions, rewards, new_states = self.memory.get_minibatch(batch_type='rand') # update to match what is obtained from get_minibatch call

		# The main network estimates which action is best (in the next state s', new_states is passed!) 
		# for every transition in the minibatch
		arg_q_max = torch.argmax(self.main_DQN.forward(new_states), dim=1)

		# The target network estimates the Q-values (in the next state s', new_states is passed!) 
		# for every transition in the minibatch
		q_vals = self.target_DQN.forward(new_states)
		# print(q_vals)
		double_q = q_vals[range(self._BATCH_SIZE_), arg_q_max]

		# Bellman equation. Multiplication with (1-terminal_flags) makes sure that 
		# if the game is over, targetQ=rewards
		target_q = rewards + self._GAMMA_*double_q

		# Q-value estimates using main DQN with action taken
		# arg_q_max = torch.argmax(self.main_DQN.forward(states), dim=1)
		est_q = self.main_DQN.forward(states)[range(self._BATCH_SIZE_), actions.tolist()]

		self.opt.zero_grad() # zeros gradient buffer to set up for next update

		# Gradient descend step to update the parameters of the main network
		loss = torch.nn.functional.smooth_l1_loss(input=est_q, target=target_q, reduction='mean')
		loss.backward() # send updates to update buffer

		self.opt.step() # take step using updates in update buffer

		return loss.item() # .item() to store as scalar value instead of tensor
