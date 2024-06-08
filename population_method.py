import torch
import numpy as np
from constants import *
from tqdm import tqdm
import copy
def population_method(policy, env, episodes, iterations, N):
    # Open the file to write the scores
    with open(PATH_SCORES_POPULATION_METHOD + '.txt', 'w') as f:
        # Best theta and score
        best_theta = list(policy.parameters())
        best_score = -float('inf')
        # tqdm is used to show a progress bar
        for _ in tqdm(range(iterations)):
            for _ in range(N):
                perturbed_theta = [param + torch.randn_like(param) for param in best_theta]
                total_score = evaluate_policy(policy, env, perturbed_theta, episodes)

                if total_score > best_score:
                    best_score = total_score
                    best_theta = perturbed_theta
                    # Load the best theta to the policy
                    for param, best_param_data in zip(policy.parameters(), best_theta):
                        param.data = best_param_data

            # Write the score to the file
            f.write(f'{total_score}\n')

def evaluate_policy(original_policy, env, theta, episodes):
    policy = copy.deepcopy(original_policy)
    # Load the perturbated parameters to the policy for evaluation
    for param, perturbed_param_data in zip(policy.parameters(), theta):
        param.data = perturbed_param_data
    
    total_score = 0
    for _ in range(episodes):
        # Reset the environment
        state = torch.tensor(env.reset()[0]).unsqueeze(0)
        done = False
        counter = 0
        while not done:
            counter += 1
            actions = policy(state).tolist()[0]
            next_step = env.step(actions)
            next_state, score, done = torch.tensor(next_step[0]).unsqueeze(0), next_step[1], next_step[2]
            total_score += score
            state = next_state
            if counter > 1000:
                break
    return total_score / episodes