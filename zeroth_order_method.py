import torch
import numpy as np
from constants import *
from tqdm import tqdm
import copy 
import os

def zeroth_order_method(policy, env, learning_rate, episodes, iterations):
    # Open the file to write the scores
    with open(PATH_SCORES_ZERO_ORDER_METHOD + '.txt', 'w') as f:
        with torch.no_grad():
            # Theta contains the parameters of the policy: {hidden.weight, hidden.bias, output.weight, output.bias}
            # Respectively, the weight of the hidden layer, the bias of the hidden layer, the weight of the output layer, and the bias of the output layer
            # tqdm is used to show a progress bar
            for _ in tqdm(range(iterations)):
                theta = list(policy.parameters())
                perturbed_theta = [torch.randn_like(param) for param in theta]
                perturbed_theta_positive = [param + perturb for param, perturb in zip(theta, perturbed_theta)]
                perturbed_theta_negative = [param - perturb for param, perturb in zip(theta, perturbed_theta)]

                total_score_positive = evaluate_policy(copy.deepcopy(policy), env, perturbed_theta_positive, episodes)
                total_score_negative = evaluate_policy(copy.deepcopy(policy), env, perturbed_theta_negative, episodes)

                gradient = [0.5 * (total_score_positive - total_score_negative) * perturb for perturb in perturbed_theta_positive]

                updated_theta = [param + learning_rate * grad for param, grad in zip(theta, gradient)]
                # Load the updated parameters to the policy
                for param, updated_param_data in zip(policy.parameters(), updated_theta):
                    param.data = updated_param_data

                # Evaluate the updated policy
                total_score = evaluate_policy(copy.deepcopy(policy), env, updated_theta, episodes)

                # Write the score to the file
                f.write(f'{total_score}\n') 


def evaluate_policy(policy, env, theta, episodes):
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
            # List of actions
            actions = policy(state).tolist()[0]
            # Take a step in the environment
            next_step = env.step(actions)
            # Get the next state, the score, and whether the episode is done
            next_state, score, done = torch.tensor(next_step[0]).unsqueeze(0), next_step[1], next_step[2]

            # Update the total score
            total_score += score

            # Update the state
            state = next_state
            # if counter > 1000:
            #     break
    return total_score / episodes


            
