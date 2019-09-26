import numpy.random as npr
from .base_agent import BaseAgent

class RandomAgent(BaseAgent):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	# determine action based on current state
	def act(self, obs=None):
		'''
		Args:
			obs is a tuple of dictionaries containing 
				'bid': bid price,
				'bid depth': total number of buy orders
				'ask': ask price
				'ask depth': total number of sell orders
				'round': n
		'''
		if obs is None:
			return {'action': npr.randint(3), 
					'price': npr.randint(95,106), 
					'volume': npr.randint(1, 11)*1000}
		else:
			# randomly act based only on current bid/ask
			obs = super().state_parser(obs)
			n = len(obs)
			if obs[n-5] is None: # bid is None
				mid = obs[n-3]
			elif obs[n-3] is None: # ask is None
				mid = obs[n-5]
			else:
				mid = (obs[n-5] + obs[n-3]) / 2

			return {'action': npr.randint(3),
					'price': mid + npr.randint(-3,4),
					'volume': npr.randint(1, 11)*1000}