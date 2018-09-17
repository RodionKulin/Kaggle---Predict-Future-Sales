import os
import sys
import time
import numpy as np
import subprocess

# Properties
runner_dir = os.getcwd()
default_experiment_executable = 'experiment.py'
shut_down_delay_mins = 5
finished_experiments = []
args = ''


# Run experiments
def get_next_experiment():
    immediate_subdirs = [name for name in os.listdir(runner_dir)
        if os.path.isdir(os.path.join(runner_dir, name))
        and name not in finished_experiments]

    if len(immediate_subdirs) > 0:
        return immediate_subdirs[0]
    else:
        return ''

def run_experiment(subdir):
    experiment_dir = os.path.join(runner_dir, subdir)
    experiment_executable = ''

    py_files = [f for f in os.listdir(experiment_dir)
                if os.path.isfile(os.path.join(experiment_dir, f))
                and f.endswith(".py")]
    if len(py_files) == 1:
        experiment_executable = os.path.join(experiment_dir, py_files[0])
    if default_experiment_executable in py_files:
        experiment_executable = os.path.join(experiment_dir, default_experiment_executable)

    try:
        print('Running experiment in directory %s' % subdir)
        sys.stdout.flush()
        if experiment_executable == '':
            raise NameError(
                'Zero or more than one python file found in experiment directory %s and default executable name %s is not used.'
                % (subdir, default_experiment_executable))
        os.chdir(experiment_dir)
        code = os.system('{} {} {}'.format(sys.executable, experiment_executable, args))
    except NameError as e:
        print(e.args[0], file=sys.stderr)
        sys.stdout.flush()
    except Exception as e:
        msg = 'Unhandled exception {}'.format(sys.exc_info()[:-1])
        print(msg, file = sys.stderr)
        sys.stdout.flush()

    finished_experiments.append(subdir)

def run_all_experiments():
    while True:
        experiment_path = get_next_experiment()
        if experiment_path == '':
            break
        run_experiment(experiment_path)


# Store arguments to pass inside experiments
def read_args():
    global args
    args = " ".join(sys.argv[1:])


# Wait for new experiments
def wait_before_shutdown():
    shutdown_delay_secs = shut_down_delay_mins * 60
    check_interval_secs = 5
    check_interval_num = shutdown_delay_secs // check_interval_secs

    for interval in range(check_interval_num):
        experiment_path = get_next_experiment()
        has_new_experiments = experiment_path != ''
        if has_new_experiments:
            return False

        time_elapsed_secs = interval * check_interval_secs
        time_left_secs = shutdown_delay_secs - time_elapsed_secs

        time_left_minutes = time_left_secs // 60
        time_left_sec = np.ceil(time_left_secs % 60)
        print('Shutting down in %d minutes %d secs' % (time_left_minutes, time_left_sec))
        sys.stdout.flush()

        time.sleep(check_interval_secs)

    print('Shutting down')
    return True

def shut_down_self():
    # Linux shut down command for EC2 instance.
    # Does not work on Windows.
    os.system('sudo halt')


# Entry point
if __name__ == "__main__":
    read_args()

    while True:
        run_all_experiments()
        time_is_up = wait_before_shutdown()
        if time_is_up:
            break

    shut_down_self()
