KDA Verification
================

## MultiBind Installation

The `verification.py` module relies on the python package
[MultiBind](https://github.com/Becksteinlab/multibind) to
get thermodynamically consistent sets of rates (at equilibrium). To install
`MultiBind`, simply follow these instructions:

```bash
1  cd ~/path/to/clone/multibind
2  git clone git@github.com:Becksteinlab/multibind.git
3  cd /multibind
4  python setup.py install
```

## Running Verification Code

### Single Dataset

To run the verification script `verification.py` and generate *root-mean-square*
(RMS) and *timing* analysis outputs, run the following on the command line:

```bash
1  cd ~/path/to/kda/scripts
2  python verification.py n_states n_runs max_rates SAVE_PATH
3  python timing.py SAVE_PATH/data
4  python rms.py SAVE_PATH/data
```

Note: the `max_rates` parameter is irrelevant if `n_states` is less than 7 but
must be specified nonetheless.

### Multiple Datasets

To get useful analysis, **line 2** must be ran for several different values
of `n_states` (i.e. 3, 4, 5, 6, 7...) and `max_rates` (more than twice the
number of states), which would look something like the
following:

```bash
1  cd ~/path/to/kda/scripts
2  python verification.py 6 100 30 SAVE_PATH
3  python verification.py 7 100 38 SAVE_PATH
4  python verification.py 8 100 38 SAVE_PATH
5  python verification.py 9 100 38 SAVE_PATH
6  python timing.py SAVE_PATH/data
7  python rms.py SAVE_PATH/data
```

Note: if no `SAVE_PATH` is specified the default is to save all files in the
**current working directory**. So an alternative approach is the following:

```bash
1  cd SAVE_PATH
2  python ~/path/to/kda/scripts/verification.py 6 100 30 ./
3  python ~/path/to/kda/scripts/verification.py 7 100 38 ./
4  python ~/path/to/kda/scripts/verification.py 8 100 38 ./
5  python ~/path/to/kda/scripts/verification.py 9 100 38 ./
6  python ~/path/to/kda/scripts/timing.py ./data
7  python ~/path/to/kda/scripts/rms.py ./data
```

### Multiple Parallel Datasets

Instead of running `verification.py` several times, the analysis can be run
using `run_verification.py`. This script is setup to essentially run batches of
2 or 3 instances of `verification.py` at a time. This requires a machine with at
least a 4 core processor, and a sufficient amount of memory to handle running
multiple datasets simultaneously.

To run `run_verification.py`, do the following:

```bash
1 cd ~/path/to/kda/scripts
2 python run_verification.py SAVE_PATH 100 5
```

The above example will run each batch of processes for a maximum of 5 hours, or
until all 100 datasets have been generated, whichever comes first. There are 5
batches in total, so the maximum run time of `run_verification.py` is 25 hours
for this case, although the first couple of batches would likely finish ahead of
time. The data will be stored at location `SAVE_PATH`. The two command line
arguments, `n_runs` and `max_run_time`, allow the user to either data-limit or
time-limit their runs, while the number of states and maximum number of rates
are hard-coded in.

## Code Outputs

### verification.py

`verification.py` creates two directories for each run case: one for the
generated `NetworkX` graphs (stored in pickle format) and another for the
`MultiBind` rates.csv files. Each **graph** and **rates** file (stored in the
corresponding directories) are indexed by the number of states and run number.
A third directory, `/data`, is created for storing all of the other pertinent
information, where a `_data.csv` file is stored for each execution of
`verification.py`. Each `_data.csv` file contains the index, number of states,
number of edges/rates, steady state probabilities, and timing information
for each run case.

### rms.py

`rms.py` uses the `_data.csv` files in `/data` to produce `data/rms_table.tex`,
a `LaTeX` table of **root-mean-square** deviations from the
**Singular Value Decomposition** solution.

### timing.py

`timing.py` uses the `_data.csv` files in `/data` to produce
`data/timing_n_nodes.pdf`, `data/timing_n_edges.pdf`,
`data/timing_avg_degree.pdf`, `data/timing_n_pars.pdf`,
`data/timing_n_dirpars_loglog.pdf`, `data/timing_n_dirpars.pdf`, and
`data/timing_n_dirpars_loglog.pdf`. If given a path to a specific `_data.csv` or
if there is only one `_data.csv` present in `/data`, it will only output
`data/timing_n_edges.pdf` and `data/timing_avg_degree.pdf`. It will also
generate a `.csv` for each **timing** plot containing the raw data used to produce
the plots, as well as an additional file `all_data.csv` which contains the data
from all of the `_data.csv` files in `/data`.
