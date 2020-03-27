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
	_MAX_PORT_SIZE_ = 50			# maximum number of shares long/short by agent
	_FINAL_EPS_ = 0.01			# long-term floor probability of taking a random action
	_MEM_START_SIZE_ = 2000	 	# number of steps of random action
	_MN_UPDATE_FREQ_ = 1		# Every C_m steps, update main network
	_TN_UPDATE_FREQ_ = 10000 	# Every C_t steps, update target network
	_BATCH_SIZE_ = 32			# batch size for batch training - equal to expected duration between dividends
	_GAMMA_ = 0.99 				# discount factor for Bellman equation
	_ALPHA_ = 0.0001 		# learning rate for parameter updates

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
		self.eval = eval
		self.prev_value	= 0
		self.assets_at_div = 0
		self.last_dividend = 0
		self.k = state_size
		if self.informed:
			self.memory = Memory(batch_size=self._BATCH_SIZE_, frame_height=2, frame_width=2)
		else:
			self.memory = Memory(batch_size=self._BATCH_SIZE_, frame_height=2, frame_width=2, window_size=10)
		if eval:
			# evaluation mode - no learning
			self.main_DQN = torch.load(self.agentId+'NN.pth')
			self.main_DQN.eval() # needed to properly load parameters
		else:
			# training mode
			if self.informed:
				self.main_DQN = DQN(n_actions=2) # action space of 30 buy/sell/nothing by duration 1 through 10
				self.target_DQN = DQN(n_actions=2)
			else:
				self.main_DQN = DQN(n_actions=2, window_size=10) # action space of 30 buy/sell/nothing by duration 1 through 10
				self.target_DQN = DQN(n_actions=2, window_size=10)
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
		# self.dividend_paid = True

	def update_val(self, dividend_amt):
		'''
		Update agent's current wealth by dividend amount
		Args:
			dividend_amt: int, amount to increase cash holdings by per unit of asset held
		'''
		self.cash += self.assets * dividend_amt
		self.dividend_paid = True
		self.assets_at_div = self.assets
		self.last_dividend = dividend_amt
		# self.assets = 0 # soft reset of state

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

		current_state = self.__state_parser(obs, price, last_executed, n)

		# update memory with new transition based on new info
		self.__update_memory(current_state, price)

		# take random action according to exploration/exploitation
		if np.abs(self.assets) < self._MAX_PORT_SIZE_:
			if np.random.random() < self.__eps_scheduler(n):
					raw_action = np.random.randint(2)
					# self.last_action = raw_action
			else:
				raw_action = self.main_DQN.act(self.memory.get_current_state()).item() # act implemented to that it assumes no exploration
				# self.last_action = raw_action.item()
		elif self.assets >= self._MAX_PORT_SIZE_:
			raw_action = 1
			# self.last_action = raw_action
		elif self.assets <= -self._MAX_PORT_SIZE_:
			raw_action = 0
		self.last_action = raw_action

		# make updates if in training mode
		if not self.eval and n > self._MEM_START_SIZE_:
			if n % self._MN_UPDATE_FREQ_ == 0:
				self.loss = self.__learn()

			# update target network as needed
			if n % self._TN_UPDATE_FREQ_ == 0:
				self.__update_target_dqn()

		# map raw action to corresponding action/duration
		# self.last_action = raw_action
		action = self.last_action # int(raw_action / 10) # 0-9: 0 (buy), 10-19: 1 (sell), 20-29: 2 (no action)
		duration = 10 # raw_action % 10 + 1 # duration in [1,10]

		if action == 0: # market buy order
			price = obs[2,0] + 10 # submit order at inside-ask + 10
		elif action == 1: # market sell order
			price = obs[0,0] - 10 # submit order at inside-bid - 10

		oId = self.agentId + '_' + str(n) # set order ID
		self.open_orders[oId] = (action, price, duration) # add to current open orders

		# for debugging
		print('state variable:\n', current_state)
		if self.informed:
			print('next dividend amount:', self.dividend_amt, 'asset holdings:', self.assets, 'action taken:', action)
		# print('agent action submitted:', oId, action, price, duration)

		return (oId, action, price, duration)

	def __state_parser(self, obs, price, last_executed, n):
		'''
		Creates current state from market observation and instance variables
		Args:
			obs: 4xk array, represents bid price/volume in decreasing order and ask price/volume in increasing order
			price: float, price at which last market cleared
			last_executed: int, volume of trades cleared at last price
			n: int, the time step
		'''
		# normalize market observation relative to mid
		inside_bid_vol = obs[1,0]
		inside_ask_vol = obs[3,0]
		if inside_bid_vol != 0 and inside_ask_vol != 0:
			mid = (obs[0,0]*inside_bid_vol + obs[2,0]*inside_ask_vol) / (inside_bid_vol + inside_ask_vol) # volume-weighted average of inside prices
		else:
			mid = (obs[0,0] + obs[2,0]) / 2

		norm_obs = np.copy(obs)
		norm_obs[0, :] = norm_obs[0, :] - mid
		norm_obs[2, :] = norm_obs[2, :] - mid
		norm_obs = norm_obs.reshape( (2,2) )

		norm_obs[0, 1] = norm_obs[0, 1] - norm_obs[1, 1] # trade imbalance
		norm_obs[1, 1] = self.assets # include number of assets somehow

		if self.informed:
			# creates addition to market state with agent-specific info and zero-pads to match length
			# addition = np.array([[price, self.cash, self.next_dividend - n], 
			# 					 [last_executed, self.assets, self.dividend_amt]])
			addition = np.array([[self.next_dividend - n], 
								 [self.dividend_amt]])
			# addition = np.pad(addition, ((0, 0), (0, self.k-3)), 'constant')

			# concatenate along horizontal axis
			# return np.vstack( (obs, addition) )

			return np.hstack( (norm_obs, addition) )[:, 1:]
		else:
			return norm_obs

	def __eps_scheduler(self, n):
		'''
		Determines epsilon at given time step
		Args:
			n: int, time period
		'''
		if n < self._MEM_START_SIZE_: # only random actions until buffer is of certain size
			return 1
		else: # linearly decreasing exploration probability with set floor
			# return max(1 - (n - self._MEM_START_SIZE_) / (5*self._MEM_START_SIZE_), self._FINAL_EPS_)
			return max(self._MEM_START_SIZE_ / n, self._FINAL_EPS_)

	def __update_memory(self, market_state, price):
		'''
		Adds previous action and resulting reward to memory, updates own state accordingly
		Args:
			market_state: 4xk array, represents state variables of market environment and agent information
			price: double, market-clearing price from last step
		'''
		new_value = self.assets*price + self.cash
		# reward = self.__clip_reward(new_value - self.prev_value) # sign of change in cumulative net P&L
		# reward = new_value - self.prev_value # change in net mark-to-market P&L
		if self.dividend_paid:
			reward = float(self.last_dividend * self.assets_at_div)
		else:
			reward = float(np.sign(new_value - self.prev_value))
			# reward = float(np.sign(self.dividend_amt * -(2*self.last_action - 1)))
		print('reward:', reward)
		self.memory.add_memory(self.last_action, market_state, reward) #, self.dividend_paid) how would taking dividend_paid work? Also need to add other things later?

		self.dividend_paid = False
		self.prev_value = new_value

	def __update_target_dqn(self):
		'''
		Updates parameters of target DQN
		'''
		self.target_DQN.load_state_dict(self.main_DQN.state_dict())

	def __clip_reward(self, reward):
		'''
		Reduces reward to sign for more stable learning
		Args:
			reward: double, value to be clipped
			rewards: numpy array, values to be clipped
		'''
		if reward > 0 and reward <= self.assets and reward >= -self.assets:
			return 1
		elif reward < 0 and reward >= -self.assets and reward <= self.assets:
			return -1
		else:
			return reward

	def __clip_discount(self, disc_vals):
		'''
		Reduces reward to sign for more stable learning
		Args:
			rewards: numpy array, values to be clipped
		'''
		for i in range(len(disc_vals)):
			if disc_vals[i] > 100:
				disc_vals[i] = 100
			elif disc_vals[i] < -100:
				disc_vals[i] = -100

		return disc_vals

	def __learn(self):
		'''
		Implements action taking, loss calculation, parameter updating
		Returns:
			loss: double for loss value
		'''
		# Draw a minibatch from the replay memory
		states, actions, rewards, new_states = self.memory.get_minibatch(batch_type='rand') # update to match what is obtained from get_minibatch call

		# The main network estimates which action is best (in the next state s', new_states is passed!) 
		# for every transition in the minibatch
		# print(self.main_DQN.forward(new_states).shape)
		arg_q_max = torch.argmax(self.main_DQN.forward(new_states), dim=1)

		# The target network estimates the Q-values (in the next state s', new_states is passed!) 
		# for every transition in the minibatch
		q_vals = self.target_DQN.forward(new_states)
		# print(q_vals)
		double_q = q_vals[range(self._BATCH_SIZE_), arg_q_max]
		# double_q = torch.max(q_vals, dim=1)[1]
		# print(double_q)

		# target Q-values from Bellman equation, clipped to be (-1, 1)
		# target_q = self.__clip_discount(rewards + self._GAMMA_*double_q)
		target_q = rewards + self._GAMMA_*double_q
		# print(target_q)

		# Q-value estimates using main DQN with action taken
		est_q = self.main_DQN.forward(states)[range(self._BATCH_SIZE_), actions.tolist()]
		# print(est_q)

		# Gradient descend step to update the parameters of the main network
		loss = torch.nn.functional.smooth_l1_loss(input=est_q, target=target_q, reduction='mean')
		# loss = torch.nn.functional.mse_loss(input=est_q, target=target_q, reduction='mean')

		self.opt.zero_grad() # zeros gradient buffer to set up for next update

		loss.backward() # send updates to update buffer

		# clip gradients?
		# for param in self.main_DQN.parameters():
		# 	param.grad.data.clamp_(-1, 1)

		# print(self.main_DQN.conv[0].weight)
		# print(self.main_DQN.linear[0].weight)
		# print(self.main_DQN.conv[0].weight.grad)

		self.opt.step() # take step using updates in update buffer

		# print(self.main_DQN.conv[0].weight)

		return loss.item() # .item() to store as scalar value instead of tensor
