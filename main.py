import gymnasium as gym
import policy as p
from constants import *
from population_method import population_method
from zeroth_order_method import zeroth_order_method
from plotting import plot_episodes, plot_total_scores, plot_total_scores_per_10, plot_total_scores_per_100, plot_combined
from utils import *
import copy
import time

def task_population_method(iteration):
    policy = p.Policy(STATE_DIM, ACTION_DIM, HIDDEN_DIM)
    env = gym.make('LunarLanderContinuous-v2')
    # population_method(policy, env, EPISODES_PM, ITERATIONS_PM, N_PM, iteration)
    save_path = f"{PATH_SCORES_POPULATION_METHOD}_{iteration}.txt"
    scores = load_list_from_file(save_path)
    plot_combined(scores, f"{PATH_PLOTS_POPULATION_METHOD}/combined/population_method_combined_{iteration}.png", EPISODES_PM)

def task_zeroth_order_method(iteration):
    policy = p.Policy(STATE_DIM, ACTION_DIM, HIDDEN_DIM)
    env = gym.make('LunarLanderContinuous-v2')
    zeroth_order_method(policy, env, LEARNING_RATE_ZOM, EPISODES_ZOM, ITERATIONS_ZOM, iteration)
    save_path = f"{PATH_SCORES_ZERO_ORDER_METHOD}_{iteration}.txt"
    scores = load_list_from_file(save_path)
    plot_combined(scores, f"{PATH_PLOTS_ZERO_ORDER_METHOD}/combined/zero_order_method_combined_{iteration}.png", EPISODES_ZOM)

def task_multithreaded():
    population_method_tasks = [(task_population_method, (i,)) for i in range(1, 12)]
    zeroth_order_method_tasks = [(task_zeroth_order_method, (i,)) for i in range(1, 12)]
    tasks = zeroth_order_method_tasks 
    run_cpu_tasks_in_parallel(tasks)  


if __name__ == '__main__':
    # The policy to be trained
    policy = p.Policy(STATE_DIM, ACTION_DIM, HIDDEN_DIM)

    # The environment
    env = gym.make('LunarLanderContinuous-v2')

    # Time the zeroth order method
    start_time = time.time()
    zeroth_order_method(copy.deepcopy(policy), env, LEARNING_RATE_ZOM, EPISODES_ZOM, ITERATIONS_ZOM, 0)
    print("--- %s seconds --- for zeroth order method" % (time.time() - start_time))

    # Load the scores from the file
    save_path = f"{PATH_SCORES}/zero_order_method.txt"
    scores = load_list_from_file(save_path)

    # Plot the scores for the zeroth order method
    plot_combined(scores, f"{PATH_PLOTS}/zero_order_method_combined.png", EPISODES_ZOM)

    # Time the population method
    start_time = time.time()
    population_method(copy.deepcopy(policy), env, EPISODES_PM, ITERATIONS_PM, N_PM, 0)
    print("--- %s seconds --- for population method" % (time.time() - start_time))

    # Load the scores from the file
    save_path = f"{PATH_SCORES}/population_method.txt"
    scores = load_list_from_file(save_path)

    # Plot the scores for the population method
    plot_combined(scores, f"{PATH_PLOTS}/population_method_combined.png", EPISODES_PM)






