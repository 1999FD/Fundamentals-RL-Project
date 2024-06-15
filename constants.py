# Policy constants (defined in project description)
STATE_DIM = 8
ACTION_DIM = 2
HIDDEN_DIM = 128

# Zeroth order method constants
EPISODES_ZOM = 5
ITERATIONS_ZOM = 1000
LEARNING_RATE_ZOM = 0.0005
# Population method constants
N_PM = 10
EPISODES_PM = 5
ITERATIONS_PM = 1000

# Paths for saving scores
PATH_SCORES = 'scores'
PATH_SCORES_POPULATION_METHOD = f'{PATH_SCORES}/population_method/population_method'
PATH_SCORES_ZERO_ORDER_METHOD = f'{PATH_SCORES}/zero_order_method/zero_order_method'

# Paths for saving plots
PATH_PLOTS = 'plots'
PATH_PLOTS_POPULATION_METHOD = f'{PATH_PLOTS}/population_method'
PATH_PLOTS_ZERO_ORDER_METHOD = f'{PATH_PLOTS}/zero_order_method'

