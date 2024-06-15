import torch
import numpy as np
from constants import *
import copy
# Parameteres: policy, env, episodes, iterations, N, i
# policy: The policy to be trained
# env: The environment
# episodes: The number of episodes to evaluate the policy
# iterations: The number of iterations to train the policy
# N: The number of perturbations to evaluate
# i: The index of the run
def population_method(policy, env, episodes, iterations, N, i):

    # Set path based on the index of the run
    # 0 is used a sequential run
    if i == 0:
        path = f"{PATH_SCORES}/population_method.txt"
    else:
        path = f"{PATH_SCORES_POPULATION_METHOD}_{i}.txt"

    # Evaluate starting policy
    total_score, episode_scores = evaluate_policy(policy, env, list(policy.parameters()), episodes)

    # Set the best score to the starting score
    best_score = total_score

    # Open the file to write the scores
    with open(path, 'w') as f:
        for iter in range(iterations):
            # Best theta and score
            best_theta = list(policy.parameters())

            # For every perturbation evaluate the policy and update the best score
            for _ in range(N):
                # Perturb the parameters
                perturbed_theta = [param + torch.randn_like(param) for param in best_theta]

                # Evaluate the perturbed policy
                total_score, episode_scores = evaluate_policy(policy, env, perturbed_theta, episodes)

                # Print total score
                print(iter, total_score, "For run: ", i)

                # Update the best score and theta if the perturbed policy is better
                if total_score > best_score:
                    best_score = total_score
                    best_theta = perturbed_theta
                    # Load the best theta to the policy
                    for (name, param), new_param in zip(policy.named_parameters(), best_theta):
                        param.data = new_param   

            # Write the score to the file
            for score in episode_scores:
                f.write(f'{score}\n')

def evaluate_policy(original_policy, env, theta, episodes):
    # Make a copy of the policy to avoid changing the original policy
    policy = copy.deepcopy(original_policy)
    for (name, param), new_param in zip(policy.named_parameters(), theta):
        param.data = new_param   
    
    # Episode scores to calculate the total score
    episode_scores = []

    # Evaluate the policy for the number of episodes
    for _ in range(episodes):

        # Reset the environment
        state = torch.tensor(env.reset()[0]).unsqueeze(0)

        # Done flag for the episode
        done = False

        # Counter for the number of steps in the episode
        counter = 0

        # Score for the episode
        episode_score = 0

        # While the episode is not done take a step in the environment
        while not done:
            # Increment the counter
            counter += 1
            actions = policy(state).tolist()[0]
            next_state, score, done, info, _ = env.step(actions)
            episode_score += score
            state = torch.tensor(next_state).unsqueeze(0)
            if counter > 1000:
                break
        episode_scores.append(episode_score)
    total_score = np.mean(episode_scores)
    return total_score, episode_scores