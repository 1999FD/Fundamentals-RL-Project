import matplotlib.pyplot as plt
from constants import PATH_PLOTS
import numpy as np

def plot_episodes(scores, path, min_value=-1000):
    # Only keep scores that are greater than min_value
    if min_value is not None:
        scores = [score for score in scores if score > min_value]
    title = path.split('/')[-1]
    plt.plot(scores)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title(f'Learning Curve {title}')
    plt.savefig(path)
    plt.close()

def plot_total_scores_per_10(scores, path, min_value=-1000):
    # Only keep scores that are greater than min_value
    if min_value is not None:
        scores = [score for score in scores if score > min_value]
    # For every 10 episodes, calculate the mean of the scores
    total_scores = [np.mean(scores[i:i+10]) for i in range(0, len(scores), 10)]
    title = path.split('/')[-1]
    plt.plot(total_scores)
    plt.xlabel('Total Score per 10 episodes')
    plt.ylabel('Total Reward')
    plt.title(f'Learning Curve {title}')
    plt.savefig(path)
    plt.close()

def plot_total_scores_per_100(scores, path, min_value=-1000):
    # Only keep scores that are greater than min_value
    if min_value is not None:
        scores = [score for score in scores if score > min_value]
    # For every 100 episodes, calculate the mean of the scores
    total_scores = [np.mean(scores[i:i+100]) for i in range(0, len(scores), 100)]
    title = path.split('/')[-1]
    plt.plot(total_scores)
    plt.xlabel('Total Score per 100 episodes')
    plt.ylabel('Total Reward')
    plt.title(f'Learning Curve {title}')
    plt.savefig(path)
    plt.close()

def plot_total_scores(scores, episodes, path, min_value=-1000):
    # Only keep scores that are greater than min_value
    if min_value is not None:
        scores = [score for score in scores if score > min_value]
    # For every x episodes, calculate the mean of the scores
    total_scores = [np.mean(scores[i:i+episodes]) for i in range(0, len(scores), episodes)]
    title = path.split('/')[-1]
    plt.plot(total_scores)
    plt.xlabel(f'Total Score per {episodes} episodes')
    plt.ylabel('Total Reward')
    plt.title(f'Learning Curve {title}')
    plt.savefig(path)
    plt.close()

# Combine all the plots in one figure
# Source: https://www.geeksforgeeks.org/plot-multiple-plots-in-matplotlib/
def plot_combined(scores, path, episodes, min_value=-1000):
    # Only keep scores that are greater than min_value
    if min_value is not None:
        scores = [score for score in scores if score > min_value]

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    # Plot episodes
    axs[0, 0].plot(scores)
    axs[0, 0].set_xlabel('Episodes', fontsize=16)
    axs[0, 0].set_ylabel('Reward', fontsize=16)
    axs[0, 0].tick_params(axis='both', which='major', labelsize=14)
    axs[0, 0].set_title('Learning Curve - Episodes', fontsize=18)

    # Plot total scores per specified number of episodes
    total_scores = [np.mean(scores[i:i+episodes]) for i in range(0, len(scores), episodes)]
    axs[0, 1].plot(total_scores)
    axs[0, 1].set_xlabel(f'Total Score per {episodes} episodes', fontsize=16)
    axs[0, 1].set_ylabel('Total Reward', fontsize=16)
    axs[0, 1].tick_params(axis='both', which='major', labelsize=14)
    axs[0, 1].set_title(f'Learning Curve - Total Scores per {episodes} Episodes', fontsize=18)

    # Plot total scores per 10 episodes
    total_scores_per_10 = [np.mean(scores[i:i+10]) for i in range(0, len(scores), 10)]
    axs[1, 0].plot(total_scores_per_10)
    axs[1, 0].set_xlabel('Total Score per 10 episodes', fontsize=16)
    axs[1, 0].set_ylabel('Total Reward', fontsize=16)
    axs[1, 0].tick_params(axis='both', which='major', labelsize=14)
    axs[1, 0].set_title('Learning Curve - Total Scores per 10 Episodes', fontsize=18)

    # Plot total scores per 100 episodes
    total_scores_per_100 = [np.mean(scores[i:i+100]) for i in range(0, len(scores), 100)]
    axs[1, 1].plot(total_scores_per_100)
    axs[1, 1].set_xlabel('Total Score per 100 episodes', fontsize=16)
    axs[1, 1].set_ylabel('Total Reward', fontsize=16)
    axs[1, 1].tick_params(axis='both', which='major', labelsize=14)
    axs[1, 1].set_title('Learning Curve - Total Scores per 100 Episodes', fontsize=18)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    
    



