import random
import matplotlib.pyplot as plt  # Import the Matplotlib library
from ice import *

# Number of episodes
EPISODES = 100_000 

# Epsilon-greedy parameters
EPSILON_START = 0.5 # Epsilon start value
EPSILON_MIN = 0.001 # Epsilon minimum value
EPSILON_DECAY_RATE = (EPSILON_MIN / EPSILON_START) ** (1 / EPISODES) # Epsilon decay rate

# Hyperparameters
GAMMA = 0.9 # Discount factor
LEARNING_RATE = 0.1 # Learning rate

def argmax(l):
    """ Return the index of the maximum element of a list """
    return max(enumerate(l), key=lambda x: x[1])[0]

def decay_epsilon(episode):
    """ Decay epsilon based on the episode number """
    return max(EPSILON_MIN, EPSILON_START * (EPSILON_DECAY_RATE ** episode))

def main():
    # Create the environment
    env = Ice()
    # Track the average cumulative reward
    average_cumulative_reward = 0.0
    # Track the cumulative rewards per episode
    cumulative_rewards = [] 

    # Track epsilon values
    epsilon_values = []

    # Q-table, 4x4 states, 4 actions per state
    qtable = [[0., 0., 0., 0.] for state in range(4 * 4)]

    # Loop over episodes
    for i in range(EPISODES):

        # Reset the environment
        state = env.reset()
        terminate = False
        cumulative_reward = 0.0

        # Decay epsilon
        epsilon = decay_epsilon(i)

        # Track epsilon values
        epsilon_values.append(epsilon)

        # Loop over time-steps
        while not terminate:
            # Compute what the greedy action for the current state is
            if random.random() < epsilon:
                a = random.randrange(4)
            else:
                a = argmax(qtable[state])

            # Compute the next state and reward
            next_state, r, terminate = env.step(a)

            # Update the Q-Table
            qtable[state][a] += LEARNING_RATE * (r + GAMMA * max(qtable[next_state]) - qtable[state][a])

            # Update statistics
            cumulative_reward += r
            state = next_state

        # Per-episode statistics
        average_cumulative_reward *= 0.95
        average_cumulative_reward += 0.05 * cumulative_reward

        # Track the average cumulative reward
        cumulative_rewards.append(average_cumulative_reward) 

    # Print the Q-table
    print("Q-table:")
    for y in range(4):
        for x in range(4):
            print('%03.3f ' % max(qtable[y * 4 + x]), end='')
        print()

    # Plot epsilon values over episodes
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(EPISODES), epsilon_values)
    plt.xlabel('Episode')
    plt.ylabel('Exploration Rate (Epsilon)')
    plt.title('Exploration Rate Decay Over Episodes')

    # Plot average cumulative reward per episode
    plt.subplot(1, 2, 2)
    plt.plot(range(EPISODES), cumulative_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Average Cumulative Reward')
    plt.title('Average Cumulative Reward Over Episodes')

    # Save the image
    plt.savefig('epsilon_and_rewards.png')

if __name__ == '__main__':
    main()
