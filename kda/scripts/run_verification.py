"""
====================
Run KDA Verification
====================
The purpose of this module is to use the functions in `verification.py` along
with Python multiprocessing to acquire multiple datasets at once. It does this
by starting up to 3 processes at once (to limit the memory footprint), running
all 3 processes for a user-specified quantity of time, then terminating the
processes and starting up new ones if the timeout is reached or if all processes
have completed. It does this for state quantities 3 - 15, in groups of 2 or 3
per collection of processes.
"""
import sys
import time
import argparse
from multiprocessing import Process

from verification import run_verification

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "save_path", type=str, nargs="?", help="Path to store output files."
    )
    parser.add_argument("n_runs", type=int, help="Number of iterations to run.")
    parser.add_argument(
        "max_run_time", type=int, nargs="?", help="Maximum run time (hours)."
    )
    args = parser.parse_args()
    save_path = args.save_path
    n_runs = args.n_runs
    max_run_time = args.max_run_time

    # here we create a dictionary for storing the parameters we want to use for
    # each run. We want to limit the number of cases running simultaneously
    # because they share memory, and some cases can use as much as 16 GB of memory
    # on their own.
    run_dict = {
        1: {
            "states": [3, 4, 5],
            "max_rates": [8, 12, 20],
        },
        2: {
            "states": [6, 7, 8],
            "max_rates": [30, 40, 40],
        },
        3: {
            "states": [9, 10, 11],
            "max_rates": [40, 40, 40],
        },
        4: {
            "states": [12, 13],
            "max_rates": [40, 40],
        },
        5: {
            "states": [14, 15],
            "max_rates": [40, 40],
        },
    }

    # convert the input to seconds
    timeout = int(max_run_time * 3600)

    for key in run_dict:
        # retrieve relevant data from dictionary
        states_list = run_dict[key]["states"]
        max_rates_list = run_dict[key]["max_rates"]
        print("=" * 40)
        print(f"Running for at most {max_run_time} hours:")
        for n_states, max_rates in list(zip(states_list, max_rates_list)):
            print(f"N states = {n_states}, maximum rates = {max_rates}")
        print("=" * 40)
        sys.stdout.flush()

        # create processes for each run
        processes = []
        for n_states, max_rates in list(zip(states_list, max_rates_list)):
            # verification_functions.run_verification() has 4 arguments:
            #       n_states, n_datasets, max_rates, and save_path
            p = Process(
                target=run_verification,
                args=(
                    n_states,
                    n_runs,
                    max_rates,
                    save_path,
                ),
            )
            p.start()
            processes.append(p)

        # initialize a time counter to check in periodically to see if
        # processes have completed
        time_counter = 0
        time_increment = 60
        while time_counter < timeout:
            # until we reach the timeout, continue
            if not any([p.is_alive() for p in processes]):
                # if all processes are dead, terminate and break
                print("All processes completed, terminating current processes...")
                sys.stdout.flush()
                for p in processes:
                    p.terminate()
                    p.join()
                break
            # if at least 1 process is alive, continue
            time.sleep(time_increment)
            time_counter += time_increment
        else:
            # if we reach the timeout, terminate
            print("Max run time reached, terminating current processes...")
            sys.stdout.flush()
            for p in processes:
                p.terminate()
                p.join()
