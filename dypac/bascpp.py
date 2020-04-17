"""
Bagging analysis of stable clusters (BASC)++
Scalable and fast ensemble clustering
"""

# Authors: Pierre Bellec, Amal Boukhdir
# License: BSD 3 clause
import warnings
from tqdm import tqdm

from scipy.sparse import csr_matrix, find
import numpy as np

from sklearn.cluster import k_means
from sklearn.preprocessing import scale

import gc

def _select_subsample(y, subsample_size, start=None):
    """Select a random subsample in a data array."""
    n_samples = y.shape[1]
    subsample_size = np.min([subsample_size, n_samples])
    max_start = n_samples - subsample_size
    if start is not None:
        start = np.min([start, max_start])
    else:
        start = np.floor((max_start + 1) * np.random.rand(1))
    stop = start + subsample_size
    samp = y[:, np.arange(int(start), int(stop))]
    return samp


def _part2onehot(part, n_clusters=0):
    """Convert a series of partition (one per row) with integer clusters into
       a series of one-hot encoding vectors (one per row and cluster).
    """
    if n_clusters == 0:
        n_clusters = np.max(part) + 1
    n_part, n_voxel = part.shape
    n_el = n_part * n_voxel
    val = np.repeat(True, n_el)
    ind_r = np.reshape(part, n_el) + np.repeat(
        np.array(range(n_part)) * n_clusters, n_voxel
    )
    ind_c = np.repeat(
        np.reshape(range(n_voxel), [1, n_voxel]), n_part, axis=0
    ).flatten()
    s_onehot = [n_part * n_clusters, n_voxel]
    onehot = csr_matrix((val, (ind_r, ind_c)), shape=s_onehot, dtype="bool")
    return onehot


def _start_window(n_time, n_replications, subsample_size):
    """Get a list of the starting points of sliding windows."""
    max_replications = n_time - subsample_size + 1
    n_replications = np.min([max_replications, n_replications])
    list_start = np.linspace(0, max_replications, n_replications)
    list_start = np.floor(list_start)
    list_start = np.unique(list_start)
    return list_start


def _trim_states(onehot, states, n_states, verbose, threshold_sim):
    """Trim the states clusters to exclude outliers."""
    for ss in tqdm(range(n_states), disable=not verbose, desc="Trimming states"):
        ix, iy, _ = find(onehot[states == ss, :])
        size_onehot = np.array(onehot[states == ss, :].sum(axis=1)).flatten()
        ref_cluster = np.array(onehot[states == ss, :].mean(dtype="float", axis=0))
        avg_stab = np.bincount(ix, weights=ref_cluster[0,iy].flatten())
        avg_stab = np.divide(avg_stab, size_onehot)
        tmp = states[states == ss]
        tmp[avg_stab < threshold_sim] = n_states
        states[states == ss] = tmp
    return states


def replicate_clusters(
    y, subsample_size, n_clusters, n_replications, max_iter=100, n_init=10, random_state=None, verbose=False, embedding=np.array([]), desc="", normalize=False
):
    """Replicate a clustering on random subsamples

    Parameters
    ----------
    y: numpy array
        size number of samples x number of features

    subsample_size: int
        The size of the subsample used to generate cluster replications

    n_replications: int
        The number of replications

    max_iter: int
        Max number of iterations for the k-means algorithm

    verbose: boolean
        Turn on/off verbose

    embedding: array
        if present, the embedding array will be appended to samp for each sample.
        For example, embedding can be a set of spatial coordinates,
        to encourage spatial proximity in the clusters.

    desc: string
        message to insert in verbose

    normalize: boolean
        turn on/off scaling of each sample to zero mean and unit variance

    Returts
    -------
    onehot: boolean, sparse array
        onehot representation of clusters, stacked over all replications.
    """
    list_start = _start_window(y.shape[1], n_replications, subsample_size)
    if (list_start.shape[0] < n_replications):
        warnings.warn("{0} replications were requested, but only {1} available.".format(n_replications, list_start.shape[0]))
    range_replication = range(list_start.shape[0])
    part = np.zeros([list_start.shape[0], y.shape[0]], dtype="int")

    for rr in tqdm(range_replication, disable=not verbose, desc=desc):
        samp = _select_subsample(y, subsample_size, list_start[rr])
        if normalize:
            samp = scale(samp, axis=1)
        if embedding.shape[0] > 0:
            samp = np.concatenate([samp, embedding], axis=1)
        _, part[rr, :], _ = k_means(
            samp,
            n_clusters=n_clusters,
            init="k-means++",
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state,
        )

    del samp
    gc.collect()
    return _part2onehot(part, n_clusters)


def find_states(onehot, n_states=10, max_iter=30, threshold_sim=0.3, n_init=10, random_state=None, verbose=False):
    """Find dynamic states based on the similarity of clusters over time."""
    if verbose:
        print("Consensus clustering.")
    _, states, _ = k_means(
        onehot,
        n_clusters=n_states,
        init="k-means++",
        max_iter=max_iter,
        random_state=random_state,
        n_init=n_init,
    )
    states = _trim_states(onehot, states, n_states, verbose, threshold_sim)
    return states


def stab_maps(onehot, states, n_replications, n_states, weights=None):
    """Generate stability maps associated with different states."""
    dwell_time = np.zeros(n_states)
    val = np.array([])
    col_ind = np.array([])
    row_ind = np.array([])

    for ss in range(0, n_states):
        if np.any(weights == None):
            dwell_time[ss] = np.sum(states == ss) / n_replications
        else:
            dwell_time[ss] = np.mean(weights[states == ss])
        if np.any(states == ss):
            stab_map = onehot[states == ss, :].mean(dtype="float", axis=0)
            mask = stab_map > 0

            row_ind = np.append(row_ind, np.repeat(ss, np.sum(mask)))
            col_ind = np.append(col_ind, np.nonzero(mask)[1])
            val = np.append(val, stab_map[mask])
    stab_maps = csr_matrix((val, (row_ind, col_ind)), shape=[n_states, onehot.shape[1]])

    # Re-order stab maps by descending dwell time
    indsort = np.argsort(-dwell_time)
    stab_maps = stab_maps[indsort, :]
    dwell_time = dwell_time[indsort]

    return stab_maps, dwell_time
