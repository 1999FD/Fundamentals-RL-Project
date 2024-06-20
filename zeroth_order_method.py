import torch
import numpy as np
from constants import *
import copy 

# Set the parameters of the policy to the new parameters theta 
def set_policy_parameters(policy, theta):
    for (name, param), new_param in zip(policy.named_parameters(), theta):
        param.data = new_param
    return policy

# Parameters : policy, env, learning_rate, episodes, iterations, i  
# policy: The policy to be trained
# env: The environment
# learning_rate: The learning rate
# episodes: The number of episodes to evaluate the policy
# iterations: The number of iterations to train the policy
# i: The index of the run
def zeroth_order_method(policy, env, learning_rate, episodes, iterations, i):
    
    # Set path based on the index of the run
    # 0 is used a sequential run
    if i == 0:
        path = f"{PATH_SCORES}/zero_order_method.txt"
    else:
        path = f"{PATH_SCORES_ZERO_ORDER_METHOD}_{i}.txt"
    
    with open(path, 'w') as f:
        for iter in range(iterations):
            # The parameters of the policy
            theta = list(policy.parameters())

            # Perturb the parameters
            perturbed_theta = [torch.randn_like(param) for param in theta]

            # Perturb the parameters in both directions
            perturbed_theta_positive = [param + perturbed_param for param, perturbed_param in zip(theta, perturbed_theta)]
            perturbed_theta_negative = [param - perturbed_param for param, perturbed_param in zip(theta, perturbed_theta)]

            # Evaluate the perturbed parameters
            total_score_positive, _ = evaluate_policy(copy.deepcopy(policy), env, perturbed_theta_positive, episodes)
            total_score_negative, _ = evaluate_policy(copy.deepcopy(policy), env, perturbed_theta_negative, episodes)

            # Calculate the gradient
            gradient = [0.5 * (total_score_positive - total_score_negative) * perturb for perturb in perturbed_theta_positive]

            # Update the parameters
            updated_theta = [param + learning_rate * grad for param, grad in zip(theta, gradient)]

            # Load the updated parameters to the policy
            policy = set_policy_parameters(policy, updated_theta)
                        
            # Evaluate the updated policy
            total_score, scores = evaluate_policy(copy.deepcopy(policy), env, updated_theta, episodes)
            # Write every score to the file
            for score in scores:
                f.write(f'{score}\n')

            # Print total score
            print(iter, total_score, "For run: ", i)



def evaluate_policy(policy, env, theta, episodes):
    # Load the perturbed parameters to the policy for evaluation
    policy = set_policy_parameters(policy, theta) 

    # List of episode scores
    episode_scores = []

    # For every episode evaluate the policy
    for _ in range(episodes):
        # Reset the environment
        state = torch.tensor(env.reset()[0]).unsqueeze(0)

        # Done flag for the episode
        done = False

        # Counter for the number of steps
        counter = 0

        # Score for the episode 
        episode_score = 0

        # While the episode is not done take a step in the environment
        while not done:
            # Increment the counter
            counter += 1

            # List of actions
            actions = policy(state).tolist()[0]

            # Take a step in the environment and get the next state, reward, done, info
            next_state, score, done, truncated, info = env.step(actions)

            # Add the reward to the episode score
            episode_score += score

            # Update the state
            state = torch.tensor(next_state).unsqueeze(0)
            
            # If the episode is too long, break
            # Normally the the LunarLander environment should end before 1000 steps, but in my case it doesn't. That is why this condition is added.
            if counter > 1000:
                done = True
        # Append the episode score to the list of episode scores
        episode_scores.append(episode_score)
    # Calculate the total score of the policy by taking the mean of the episode scores
    total_score = np.mean(episode_scores)

    return total_score, episode_scores



            
