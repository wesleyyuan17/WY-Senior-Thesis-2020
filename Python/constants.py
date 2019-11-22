# action mappings
BUY_ORDER = 0
SELL_ORDER = 1
NO_ACTION = 2

# set run parameters
DEBUG = True # True for testing
if DEBUG:
	DEBUG_ROUNDS = 100000
NUM_TRAINED_AGENTS = 2
PERCENT_INFORMED = 0.5
NUM_NOISE_AGENTS = 20
NOISE_ACTION_SPACE = 300

# market parameters
DIVIDEND_PROB = 0.2 # change distribution of dividend dates
DIVIDEND_MEAN = 0 # change distribution of dividend amount, maybe go for fat tailed?
DIVIDEND_STD = 1
MARKET_DEPTH = 10

# control training parameters
BATCH_SIZE = 20 # number of states returned per training cycle
BATCH_WINDOW = 3 # number of consecutive market states that make up input state