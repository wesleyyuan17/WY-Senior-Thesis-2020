# action mappings
BUY_ORDER = 0
SELL_ORDER = 1
NO_ACTION = 2

# set run parameters
DEBUG = True # True for testing
if DEBUG:
	DEBUG_ROUNDS = 10000
NUM_AGENTS = 200

# Dividend parameters
CONCURRENT_DIVIDEND = False # overlapping dividends?
DIVIDEND_PROB = 0.2 # change distribution of dividend dates
DIVIDEND_MEAN = 0 # change distribution of dividend amount, maybe go for fat tailed?
DIVIDEND_STD = 1

# control training parameters
BATCH_SIZE = 20 # number of states returned per training cycle
BATCH_WINDOW = 3 # number of consecutive market states that make up input state