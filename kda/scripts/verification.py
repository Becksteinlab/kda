"""
================
KDA Verification
================
The purpose of this module is to test the outputs of Kinetic Diagram Analysis
against the matrix solution. It does this by generating random sets of rate
constants, using MultiBind to get a thermodynamically consistent set of
equilibrium rate constants, then running them through KDA state probability
calculation functions `svd.matrix_solver()` and
`calculations.calc_state_probs()`. A host of data is produced and stored in
CSV files for analysis via modules `timing.py` and `rms.py`, and all
produced rates and graphs are stored in corresponding directories.
"""
import os
from os.path import join
import sys
import argparse
import time
from tqdm import tqdm

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import networkx as nx

from kda import calculations, diagrams, graph_utils, svd
from multibind import nonequilibrium


class HiddenPrints:
    """
    Class used to suppress unwanted print statements.
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def get_off_diagonal_indices(arr):
    """
    Takes an `NxN` matrix as input and returns the indices of the off-diagonal
    array elements.
    """
    # get the number of nodes
    N = arr.shape[0]
    # get the indices for the diagonals (main diagonal is 0)
    diag_indices = np.arange(N)
    # remove the main diagonal, first off diagonal, and furthest diagonal
    mask1 = diag_indices != 0
    mask2 = diag_indices != 1
    mask3 = diag_indices != N - 1
    mask = mask1 & mask2 & mask3
    diag_indices = diag_indices[mask]
    # now that we know which diagonals we can set equal to zero, let's
    # get the indices of the elements in those off-diagonals
    off_diagonal_indices = []
    for idx in diag_indices:
        # create a mask of identical size to input array
        diag_mask = np.diag(np.ones(N - idx, dtype=bool), k=idx)
        # get the indices of the off-diagonal elements
        indices = np.column_stack((np.nonzero(diag_mask)))
        # append each pair
        for pair in indices:
            off_diagonal_indices.append(pair)
    return np.asarray(off_diagonal_indices)


def get_min_connections(N, max_rates):
    """
    Returns the minimum number of connections based on the number of states in
    a diagram. Keeping too many connections results in a large memory footprint
    that scales as roughly N!, so we have to arbitrarily limit the number of
    connections in a diagram.
    """
    # the minimum number of rates is twice the number of states
    if max_rates <= (2 * N):
        raise ValueError(
            f"Not enough rates to generate diagram. Set `max_rates`"
            f" parameter to more than twice the number of states."
        )
    # for smaller input diagrams we don't have to arbitrarily limit the
    # number of edges
    if N <= 6:
        return 0
    # for larger state quantities we use `max_rates` to arbitrarily limit
    # the number of connections in the diagram
    else:
        max_cons_in_diagram = (N * (N - 1)) / 2
        max_cons = int(max_rates / 2)
        min_cons_to_mute = max_cons_in_diagram - max_cons
        return min_cons_to_mute


def construct_varied_array(N, arr, off_diag_idx, min_to_mute, max_rates):
    """
    Takes an `NxN` array with all valid connections as input and returns
    an `NxN` array with some connections muted.
    """
    # for 3-state models there are no connections to mute
    if N == 3:
        return arr
    # calculate the maximum number of connections that can be muted
    max_to_mute = (N ** 2 - 3 * N) / 2
    # generate a random number that determines the
    # number of connections to mute in the array
    n_cons_to_mute = np.random.randint(low=min_to_mute, high=max_to_mute + 1)
    # since the random numbers generated are in a narrow range, we need to
    # generate many more indices to make sure we get enough unique indices to mute
    indices_to_mute_size = n_cons_to_mute * N ** 2
    # generate a random selection of indices to mute
    indices_to_mute = np.random.randint(
        low=0, high=len(off_diag_idx), size=indices_to_mute_size
    )
    # take the first n_cons_to_mute unique indices
    indices_to_mute = np.unique(indices_to_mute)
    indices_to_mute = indices_to_mute[:n_cons_to_mute]
    # check that we have the correct number of indices
    if indices_to_mute.size != n_cons_to_mute:
        raise ValueError(
            f"Not enough randomly generated connections. Wanted"
            f" to mute {n_cons_to_mute} connections, only collected"
            f" {indices_to_mute.size} connections."
        )
    # keep the selected connections/rate-pairs
    muted_indices = off_diag_idx[indices_to_mute]
    # generate a new array to return
    new_array = arr.copy()
    # for each pair of indices kept in indices to mute, set each value to zero
    for idx in muted_indices:
        new_array[idx[0], idx[1]] = 0
        new_array[idx[1], idx[0]] = 0
    n_rates = np.nonzero(new_array)[0].size
    if n_rates > max_rates:
        raise ValueError(
            f"Too many rates kept in diagram. Wanted no more than"
            f" {max_rates} rates, collected {n_rates} rates."
        )
    return new_array


def write_multibind_rates_file(rate_matrix, rates_path):
    """
    Function for converting KDA rate matrix into MultiBind form and storing the
    relevant data in a CSV file.
    """
    # get data for rates.csv
    s1_idx, s2_idx = np.nonzero(rate_matrix)
    k_vals = rate_matrix[s1_idx, s2_idx]
    state1 = []
    state2 = []
    for s1, s2 in list(zip(s1_idx, s2_idx)):
        state1.append(f"s{s1}")
        state2.append(f"s{s2}")
    random_vars = 0.5 * np.random.random(size=(k_vals.shape[0]))
    k_vars = k_vals * random_vars
    # create pandas dataframes to save
    rates_cols = ["state1", "state2", "k", "kvar"]
    rates_data = list(zip(state1, state2, k_vals, k_vars,))
    rates_df = pd.DataFrame(data=rates_data, columns=rates_cols)
    # save dataframes as .csv file at location rates_path
    rates_df.to_csv(path_or_buf=rates_path, sep=",", columns=rates_cols, index=False)


def get_thermodynamically_consistent_matrix(K0, rates_save_path, index):
    """
    Function for handling host of functions for converting an input rate matrix
    into a thermodynamically consistent matrix.
    """
    # make the paths for each MultiBind .csv file
    n_states = K0.shape[0]
    rates_path = join(rates_save_path, f"{index}_rates.csv")
    # use input K matrix to write rates.csv files in MultiBind format
    write_multibind_rates_file(rate_matrix=K0, rates_path=rates_path)
    # use MultiBind to get a new set of rates that are consistent with
    # thermodynamics
    K_consistent = nonequilibrium.rate_matrix(rates_path)[1]
    # overwrite original rates file with thermodynamically consistent rates.csv
    write_multibind_rates_file(rate_matrix=K_consistent, rates_path=rates_path)
    return K_consistent


def check_diagram_counts(G, K, dirpar_edges, n_states):
    """
    Uses KDA function to calculate the theoretical number of partial diagrams
    and verifies that it agrees with the number of partial diagrams generated
    for the state probability calculations. Also verifies that the number of
    generated directional partial diagrams agree with the theoretical value.
    """
    n_pars_theoretical = diagrams.enumerate_partial_diagrams(K0=K)
    # generate the partial diagrams for counting purposes
    n_pars = len(diagrams.generate_partial_diagrams(G))
    # assert that the calculated value is equal to the experimental value
    if n_pars_theoretical != n_pars:
        raise AssertionError(
            f"Number of partial diagrams do not match. "
            f"Expected {n_pars_theoretical} partials, generated {n_pars} partials."
        )
    n_dirpars = len(dirpar_edges)
    # check that the number of directional partial diagrams are equal to
    # the number of states times the number of partial diagrams
    assert n_dirpars == n_states * n_pars
    return n_pars, n_dirpars


def check_cycles(G, unique_cycles):
    """
    Checks that there is at least 1 cycle that contains all nodes (Hamiltonian
    cycle) in the randomly generated diagram of interest. Also verifies that
    there is at least 1 unique cycle in the input diagram.
    """
    # get a list of the nodes
    node_list = list(G.nodes)
    # initialize a variable assuming the diagram does not have a cycle that
    # contains all nodes
    contains_all_nodes = False
    # iterate over unique cycles
    for cycle in unique_cycles:
        if all(n in node_list for n in cycle):
            # if a cycle contains all nodes, mark diagram as valid
            contains_all_nodes = True
    # assert that at least one cycle contains all nodes
    assert len(unique_cycles) >= 1
    assert contains_all_nodes == True


def check_net_cycle_fluxes(G, dirpar_edges, unique_cycles):
    """
    Verifies that for every cycle in the generated diagram, the calculated net
    cycle flux is very close to zero. Since MultiBind returns a set of rates
    that are at equilibrium, there should be a zero net cycle flux for every
    cycle.
    """
    with HiddenPrints():
        # in order to speed things up we will manually calculate the net
        # cycle fluxes instead of using the built-in KDA function
        # calculations.calc_net_cycle_flux() (so we don't have to calculate
        # sigma every time)
        sigma = calculations.calc_sigma(G, dirpar_edges, key="val")

        for cycle in unique_cycles:
            # normally one has to manually determine the order of the nodes that
            # is considered the "positive" flux direction for a given physiological
            # function to make sure the net cycle flux calculation yields the
            # correct sign, but since we aren't using the net cycle fluxes
            # we can just assign an arbitrary direction and not worry about signs
            order = cycle[:2]
            flux_diags = diagrams.generate_flux_diagrams(G, cycle)
            pi_diff = calculations.calc_pi_difference(G, cycle, order, key="val",)
            sigma_K = calculations.calc_sigma_K(G, cycle, flux_diags, key="val")
            net_cycle_flux = pi_diff * sigma_K / sigma

            if not np.isclose(net_cycle_flux, 0, rtol=1e-10, atol=1e-12):
                # since the MultiBind thermodynamic consistency conversion
                # puts everything in equilibrium, we should get a zero net
                # cycle flux for all cycles
                raise ValueError(
                    f"Calculated a non-zero net cycle flux for cycle: {cycle} "
                    f"Net cycle flux: {net_cycle_flux}"
                )
    return len(unique_cycles)


def check_transition_fluxes(G, prob_arr):
    """
    To the same end as `check_net_cycle_fluxes()`, this verifies the net
    transition fluxes for each pair of states is very close to zero.
    This is done by finding all the unique connections in the diagram G,
    calculating the transition flux in each direction (J_ij and J_ji), then
    checking that they are equal. If they are all equal, then
    J_ij - J_ji = 0 for all i, j, which means we have a set of equilibrium rates.
    """
    unique_edges = diagrams._find_unique_edges(G)
    for edge in unique_edges:
        # assign the tuple values to i and j for readability
        i = edge[0]
        j = edge[1]
        # get the values for the edges from G
        kij = G[i][j][0]["val"]
        kji = G[j][i][0]["val"]
        # get the state probabilities for nodes i and j
        pi = prob_arr[i]
        pj = prob_arr[j]
        # calculate the transition fluxes in both directions
        J_ij = pi * kij
        J_ji = pj * kji
        assert_allclose(J_ij, J_ji, rtol=1e-8, atol=1e-11)


def get_paths(save_path, n_states):
    """
    Function for creating save paths for various output data.
    """
    # get current working directory
    cwd = os.getcwd()
    if save_path is None:
        print(f"No save path specified. Setting save path to current working directory")
        save_path = cwd

    os.chdir(save_path)
    # create a new directory to save all NetWorkX graphs
    graph_dir = f"{n_states}_state_graphs"
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)

    # create a new directory to save all MultiBind rates.csv files
    rates_dir = f"{n_states}_state_rates"
    if not os.path.exists(rates_dir):
        os.makedirs(rates_dir)

    # create a new directory to save all data.csv files
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    graph_save_path = join(save_path, graph_dir)
    rates_save_path = join(save_path, rates_dir)
    data_save_path = join(save_path, data_dir)

    return graph_save_path, rates_save_path, data_save_path


def construct_rate_matrix(
    n_states, min_cons_to_mute, max_rates, rates_save_path, graph_index
):
    # start by generating random data set
    powers = 5 * (np.random.random(size=(n_states, n_states)) - 0.5)
    dataset = 10 ** powers
    # now we have to convert these into NxN K-matrices, where `N` is the number
    # of states in the diagram
    K_matrix = np.reshape(dataset, newshape=(n_states, n_states))
    # fill the diagonal elements with zero values
    np.fill_diagonal(K_matrix, val=0)
    # get the indices for the off-diagonal elements
    off_diagonal_indices = get_off_diagonal_indices(K_matrix)
    # create new array with random muted rates
    K = construct_varied_array(
        N=n_states,
        arr=K_matrix,
        off_diag_idx=off_diagonal_indices,
        min_to_mute=min_cons_to_mute,
        max_rates=max_rates,
    )
    # run K-matrix through MultiBind thermodynamic consistency function
    K = get_thermodynamically_consistent_matrix(
        K0=K, rates_save_path=rates_save_path, index=graph_index,
    )
    return K


def main(
    n_states, n_datasets, max_rates, save_path,
):
    """
    Primary function for running verification.
    """
    if max_rates % 2 != 0:
        print(f"`max_rates` not evenly divisible by 2. Increasing `max_rates` by 1.")
        max_rates += 1

    graph_save_path, rates_save_path, data_save_path = get_paths(
        save_path=save_path, n_states=n_states
    )

    print(f"Saving NetWorkX graph files at location: {graph_save_path}")
    print(f"Saving MultiBind rates.csv files at location: {rates_save_path}")
    print(
        f"Generating {n_datasets} {n_states}-state models with no more than"
        f" {int(max_rates/2)} connections for verification..."
    )

    # get the minimum number of connections to mute in varied array
    min_cons_to_mute = get_min_connections(N=n_states, max_rates=max_rates)

    # create empty arrays/lists for storing data
    mat_data = np.zeros((n_datasets, n_states))
    kda_data = np.zeros((n_datasets, n_states))
    mat_time = np.zeros(n_datasets)
    kda_time = np.zeros(n_datasets)
    graph_edge_count = np.zeros(n_datasets)
    graph_cycle_count = np.zeros(n_datasets)
    dirpar_count = np.zeros(n_datasets)
    par_count = np.zeros(n_datasets)
    graph_indices = []
    for i in tqdm(range(n_datasets), desc="Models", file=sys.stdout):
        # create an index for each graph for identification purposes
        graph_index = f"{n_states}_{i+1}"
        while True:
            # construct a thermodynamically consistent rate matrix
            # from randomly generated numbers
            K = construct_rate_matrix(
                n_states=n_states,
                min_cons_to_mute=min_cons_to_mute,
                max_rates=max_rates,
                rates_save_path=rates_save_path,
                graph_index=graph_index,
            )
            # arbitrarily limit the min and max values to stay within
            # 1e-7 and 1e7 to ensure calculation accuracy (for fair comparison)
            K_min = K[K != 0].min()
            K_max = K[K != 0].max()
            if (K_min > 1e-7) and (K_max < 1e7):
                # if the min/max values are within
                # the desired range, continue
                break

        # get the matrix solution
        mat_start = time.perf_counter()
        mat_probs = svd.matrix_solver(K)
        mat_elapsed = time.perf_counter() - mat_start

        # initialize an empty graph object
        G = nx.MultiDiGraph()
        # get KDA solution
        kda_start = time.perf_counter()
        # use KDA to generate the edges from the rate matrix
        graph_utils.generate_edges(G, K, names=None)
        # generate the directional partial diagrams for calculations
        dirpar_edges = diagrams.generate_directional_partial_diagrams(
            G, return_edges=True
        )
        # use input diagram and directional partial diagrams to calculate
        # the state probabilities
        kda_probs = calculations.calc_state_probs_from_diags(G, dirpar_edges, key="val")
        kda_elapsed = time.perf_counter() - kda_start

        # hide prints from KDA function
        with HiddenPrints():
            # get all unique cycles in G
            unique_cycles = graph_utils.find_all_unique_cycles(G)
        # perform various checks and acquire remaining data for G
        check_cycles(G=G, unique_cycles=unique_cycles)
        par_diag_count, dir_par_diag_count = check_diagram_counts(
            G=G, K=K, dirpar_edges=dirpar_edges, n_states=n_states,
        )
        n_cycles = check_net_cycle_fluxes(
            G=G, dirpar_edges=dirpar_edges, unique_cycles=unique_cycles
        )
        # verify that the state probabilities are normalized to 1
        assert np.isclose(np.sum(kda_probs), 1.0, rtol=1e-5, atol=1e-8)
        assert np.isclose(np.sum(mat_probs), 1.0, rtol=1e-5, atol=1e-8)
        # check the transition fluxes for both KDA and the matrix solution
        check_transition_fluxes(G=G, prob_arr=kda_probs)
        check_transition_fluxes(G=G, prob_arr=mat_probs)

        # pickle and save the graph
        graph_save_string = f"graph_{n_states}_{i+1}.pk"
        nx.write_gpickle(G, join(graph_save_path, graph_save_string))

        # store/assign relevant data
        mat_data[i] = mat_probs
        kda_data[i] = kda_probs
        mat_time[i] = mat_elapsed
        kda_time[i] = kda_elapsed
        dirpar_count[i] = dir_par_diag_count
        par_count[i] = par_diag_count
        graph_edge_count[i] = int(G.number_of_edges())
        graph_cycle_count[i] = n_cycles
        graph_indices.append(graph_index)

    # generate array for number of states
    graph_node_count = n_states * np.ones(n_datasets, dtype=int)

    # create column headings for dataframe
    cols = [
        "graph index",
        "n_states",
        "n_edges",
        "n_cycles",
        "n_pars",
        "n_dirpars",
    ]
    time_cols = ["mat time (s)", "kda time (s)"]
    cols.extend(time_cols)
    mat_prob_cols = [f"mat p_{i+1}" for i in range(n_states)]
    cols.extend(mat_prob_cols)
    kda_prob_cols = [f"kda p_{i+1}" for i in range(n_states)]
    cols.extend(kda_prob_cols)

    # collect data for dataframe
    data = [
        graph_indices,
        graph_node_count,
        graph_edge_count,
        graph_cycle_count,
        par_count,
        dirpar_count,
    ]
    data.extend([mat_time, kda_time])
    mat_probs = [mat_data[:, i] for i in range(n_states)]
    data.extend(mat_probs)
    kda_probs = [kda_data[:, i] for i in range(n_states)]
    data.extend(kda_probs)
    data = np.array(data).T

    # create and save dataframe as .csv
    df = pd.DataFrame(data=data, columns=cols)
    print(f"Saving data.csv at location: {data_save_path}")
    csv_save_string = join(data_save_path, f"{n_states}_state_data.csv")
    df.to_csv(path_or_buf=csv_save_string, sep=",", columns=cols, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n_states", type=int, help="Number of states to run.")
    parser.add_argument("n_runs", type=int, help="Number of iterations to run.")
    parser.add_argument(
        "max_rates",
        type=int,
        help="Max number of rates in each diagram. Set to more than twice the number of states.",
    )
    parser.add_argument(
        "save_path", type=str, nargs="?", help="Path to store output files."
    )
    args = parser.parse_args()
    n_states = args.n_states
    n_datasets = args.n_runs
    max_rates = args.max_rates
    save_path = args.save_path

    # run main function that handles verification
    main(
        n_states=n_states,
        n_datasets=n_datasets,
        max_rates=max_rates,
        save_path=save_path,
    )
