#!/usr/bin/python3
from concurrent.futures import ProcessPoolExecutor
import os
import numpy as np
import matplotlib.pyplot as plt
import bandit
import plot

# Epsilon Greedy Algorithm with Decay Rate for Exploration-Exploitation Tradeoff
def run_experiment(init_epsilon, min_epsilon=0.00001, total_timesteps=900_000):

    # The decay rate for epsilon
    # The epsilon value decays exponentially over time to encourage more exploitation than exploration as the number of timesteps increases 
    # The decay rate is calculated using the formula: decay_rate = (min_epsilon / init_epsilon) ^ (1 / total_timesteps)
    decay_rate = (min_epsilon / init_epsilon) ** (1 / total_timesteps)

    # The bandit problem
    b = bandit.Bandit()

    # The cumulative regret
    cumulative_regret = 0.0

    # The Q values for each arm
    Q = np.zeros(b.num_arms())

    # The number of selections for each arm
    N = np.zeros(b.num_arms())

    # The cumulative regret at each timestep
    regrets = []
    
    for timestep in range(total_timesteps):
        # Update epsilon value
        epsilon = max(min_epsilon, init_epsilon * (decay_rate ** timestep))

        # Depending on the random value either choose a random arm or the best arm
        if np.random.rand() < epsilon:
            a = np.random.randint(b.num_arms())
        else:
            max_Q = np.max(Q)
            max_indices = np.where(Q == max_Q)[0]
            a = np.random.choice(max_indices)

        # The reward for the chosen arm
        R = b.trigger(a)

        # Update the number of selections and Q values for the chosen arm
        N[a] += 1
        Q[a] += (R - Q[a]) / N[a]

        # Update the cumulative regret
        cumulative_regret += b.opt() - R

        # Store the cumulative regret at each timestep
        regrets.append(cumulative_regret)

    # Plot the cumulative regret and selections count of each arm
    plot.plot_cumulative_regret(regrets, N, init_epsilon)

if __name__ == '__main__':
    # Run the experiment for different epsilon values
    epsilon_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        executor.map(run_experiment, epsilon_values)
