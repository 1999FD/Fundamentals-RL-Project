import matplotlib.pyplot as plt
from constants import PATH_PLOTS

def plot_scores(scores, path):
    title = path.split('/')[-1]
    plt.plot(scores)
    plt.xlabel('Epsidoes')
    plt.ylabel('Reward')
    plt.title(f'Learning Curve {title}')
    plt.savefig(path)
    plt.close()

