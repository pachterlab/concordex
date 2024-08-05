from sklearn.neighbors import NearestNeighbors
import numpy as np

def consolidate(g, labels):
    """
    Compute the neighborhood consolidation matrix. 

    g: array-like of shape (n_samples, n_neighbors) 
        A numeric matrix-like object containing indicies for neighbors 
        on the columns and observations (e.g. cells/spots) on the rows.
    
    labels: array-like
        Observation labels matrix of shape used to compute the neighborhood consolidation
        matrix. Continuous or discrete labels are allowed, and typically, integer
        labels are assumed to be discrete.

    """
    n_obs, n_neighbors = g.shape
    n_label_dim = labels.shape[1]

    # Initialize an empty result array
    result = np.zeros((n_obs, n_label_dim))

    # Flatten indices_array to apply advanced indexing in a vectorized way
    flat_indices = g.ravel()

    # Create a repeated row index array corresponding to the flat_indices
    repeat_index = np.repeat(np.arange(n_obs), n_neighbors)

    # Accumulate sums using np.add.at
    np.add.at(result, repeat_index, labels[flat_indices])

    return result / n_neighbors

def findKNN(x, n_neighbors=30, **kwargs):
    """
    A thin wrapper around sklearn NearestNeighbors. Updates to remove self-
    referential neighbors
    n_neighbors : int, default=30
        Number of neighbors used to compute the kNN graph. Defaults to 30.

    **kwargs : dict
        Additional keyword arguments passed to NearestNeighbors
    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, **kwargs)
    nbrs.fit(x)

    g = nbrs.kneighbors(x, return_distance=False)

    # Remove self from neighbors list
    return g[:,1:]

