from constants import *
from concurrent.futures import ProcessPoolExecutor

# Load list of values from a file
def load_list_from_file(filename):
    with open(filename, 'r') as f:
        return [float(line.rstrip()) for line in f]

# Run tasks in parallel on the CPU
# Source: https://docs.python.org/3/library/concurrent.futures.html
def run_cpu_tasks_in_parallel(tasks):
    with ProcessPoolExecutor() as executor:
        running_tasks = [executor.submit(task_func, *task_args) for task_func, task_args in tasks]
        for running_task in running_tasks:
            running_task.result()






