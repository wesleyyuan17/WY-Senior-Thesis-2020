import numpy.random as npr
from .base_agent import BaseAgent

class RandomAgent(BaseAgent):
	def __init__(self, *args, **kwargs):
		super(RandomAgent, self).__init__(*args, **kwargs)

	def act(self, obs):
		if obs['bid'] or obs['ask'] is not None:
			if obs['bid'] is None:
				mid = obs['ask']
			elif obs['ask'] is None:
				mid = obs['bid']
			else:
				mid = 0.5*(obs['bid'] + obs['ask'])
			return {'action': npr.randint(3), 
					'price': mid + npr.randint(-3,4), 
					'volume': npr.randint(1,11)*1000}
		else:
			return {'action': npr.randint(3), 
					'price': npr.randint(95,106), 
					'volume': npr.randint(1, 11)*1000}