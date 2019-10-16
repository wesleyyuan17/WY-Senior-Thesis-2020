def dict_to_list(d, bid=True):
	'''
	Function that converts a dictionary of price, volume pairs and returns each in its own list
	Args:
		d: dictionary, contains price, volume pairs with price as key
		bid: bool, is the dict being passed of bid price/volume
	Returns:
		2 lists, one with all prices, one with all volumes, same index are associated
	'''
	prices = []
	volumes = []
	for k,v in d.items():
		prices.append(k)
		volumes.append(v)

	return prices.sort(reverse=bid), volumes.sort(reverse=bid)

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