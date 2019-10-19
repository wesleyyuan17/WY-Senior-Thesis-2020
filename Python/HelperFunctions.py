import numpy as np

def dict_to_list(d, bid=True):
	'''
	Function that converts a dictionary of price, volume pairs and returns each in its own list
	Args:
		d: dictionary, contains price, volume pairs with price as key
		bid: bool, is the dict being passed of bid price/volume
	Returns:
		2 lists, one with all prices, one with all volumes, same index are associated
	'''
	price_volume = []
	for k, v in d.items():
		price_volume.append( (k, len(v)) )

	price_volume.sort(key=lambda x: x[0], reverse=bid)

	return price_volume

def order_book_to_market(bid_px_vol, ask_px_vol, n):
	'''
	Turns current limit order book into a state of market of 4 n-vectors
	Args:
		bid_px_vol: list, list of tuples of current bid price/volumes in descending order
		ask_px_vol: list, list of tuples of current ask price/volumes in descending order
		n: int, how deep into orders on each side to go to create market state (0 pad if less than n)
	Returns:
		market state as a 4xn numpy array
	'''
	bid_px_vol_array = np.array(bid_px_vol)
	nrow = bid_px_vol_array.shape[0]
	if len(bid_px_vol_array) > 0:
		bid_px_vol_array = np.pad(bid_px_vol_array, ((0, max(0, 10-nrow)), (0,0)), 'constant')
	else:
		bid_px_vol_array = np.zeros(shape=(10,2))

	ask_px_vol_array = np.array(ask_px_vol)
	nrow = ask_px_vol_array.shape[0]
	if len(ask_px_vol_array) > 0:
		ask_px_vol_array = np.pad(ask_px_vol_array, ((0, max(0, 10-nrow)), (0,0)), 'constant')
	else:
		ask_px_vol_array = np.zeros(shape=(10,2))

	return np.hstack( (bid_px_vol_array[:10, :], ask_px_vol_array[:10,:]) ).T


def fill_orders(bid_px, bid_vol, ask_px, ask_vol):
	'''
	Takes bid/ask prices/volumes and matches until mid price is reached
	Args:
		bid_px: list, bid prices sorted in descending order
		bid_vol: list, corresponding volume for each bid price level
		ask_px: list, ask prices sorted in ascending order
		ask_vol: list, corresponding volume for each ask price level
	'''
	while len(bid_px) > 0 and len(ask_px) > 0 and bid_px[0] >= ask_px[0]:
		top_bid_vol = bid_vol[0] 			# volume of highest bid price
		top_ask_vol = ask_vol[0] 			# volume of lowest ask price
		if top_bid_vol < top_ask_vol[0]: 	# all of top bids are filled
			ask_vol[0] -= top_bid_vol 		# update unfilled volume
			bid_px.pop() 					# delete top bid order from price/volume lists
			bid_vol.pop()
		elif top_bid_vol > top_ask_vol: 	# all of top asks are filled
			bid_vol[0] -= top_ask_vol 		# update unfilled volume
			ask_px.pop() 					# delete lowest ask order from price/volume lists
			ask_vol.pop()
		else: 								# bid/ask perfectly cancel out
			bid_px.pop()					# delete first order from all bid/ask price/volume lists
			bid_vol.pop()
			ask_px.pop()
			ask_vol.pop()