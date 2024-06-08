import gymnasium as gym
import policy as p
from constants import *
from population_method import population_method
from zeroth_order_method import zeroth_order_method
from plotting import plot_scores
from utils import *
import copy

def task_population_method(policy, env):
    population_method(copy.deepcopy(policy), env, EPISODES, ITERATIONS, N)

def task_zeroth_order_method(policy, env):
    zeroth_order_method(copy.deepcopy(policy), env, LEARNING_RATE, EPISODES, ITERATIONS)

if __name__ == '__main__':
    policy = p.Policy(STATE_DIM, ACTION_DIM, HIDDEN_DIM)
    env = gym.make('LunarLanderContinuous-v2')
    tasks = [
        (task_population_method, (policy, env)), 
        (task_zeroth_order_method, (policy, env))
        ]
    run_cpu_tasks_in_parallel(tasks)
    scores_population_method = load_list_from_file(PATH_SCORES_POPULATION_METHOD)
    plot_scores(scores_population_method, PATH_PLOTS_POPULATION_METHOD)
    scores_zero_order_method = load_list_from_file(PATH_SCORES_ZERO_ORDER_METHOD)
    plot_scores(scores_zero_order_method, PATH_PLOTS_ZERO_ORDER_METHOD)



