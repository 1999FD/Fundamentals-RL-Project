from constants import *
import torch
from concurrent.futures import ProcessPoolExecutor
from plotting import plot_scores

# Load list of values from a file
def load_list_from_file(filename):
    with open(filename + '.txt', 'r') as f:
        return [float(line.rstrip()) for line in f]
    
def get_perturbation(theta):
    perturbation = {}
    for key, value in theta.items():
        perturbation[key] = value + torch.randn_like(value)
    return perturbation

def get_theta_positive(theta, perturbation):
    theta_positive = {}
    for key, value in theta.items():
        theta_positive[key] = value + perturbation[key]
    return theta_positive

def get_theta_negative(theta, perturbation):
    theta_negative = {}
    for key, value in theta.items():
        theta_negative[key] = value - perturbation[key]
    return theta_negative

def run_cpu_tasks_in_parallel(tasks):
    with ProcessPoolExecutor() as executor:
        running_tasks = [executor.submit(task_func, *task_args) for task_func, task_args in tasks]
        for running_task in running_tasks:
            running_task.result()






