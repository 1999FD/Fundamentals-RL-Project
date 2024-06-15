import numpy as np
import matplotlib.pyplot as plt

def plot_cumulative_regret(regrets, selections, epsilon):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # The first subplot shows the cumulative regret over time
    ax1.plot(regrets, label=f'Cumulative Regret (ε={epsilon})')
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Cumulative Regret')
    ax1.set_title('Cumulative Regret over Time')
    ax1.legend()
    ax1.grid(True)

    # The second subplot shows the selections count of each arm at the end of the experiment
    arms = np.arange(len(selections)) 
    ax2.bar(arms, selections, color='red', label='Selections')
    ax2.set_xlabel('Arm Index')
    ax2.set_ylabel('Selections Count')
    ax2.set_title('Arms Selections')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f'plots/epsilon_{epsilon}.png')
    plt.close()