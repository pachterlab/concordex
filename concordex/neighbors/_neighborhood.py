import numpy as np
import warnings 

from anndata import AnnData
from sklearn.neighbors import NearestNeighbors

from ..utils._labels import Labels

def consolidate(
    adata: AnnData, 
    labels, 
    *, 
    compute_similarity: bool = False,
    key_added: str | None = None, 
    copy: bool = False
):
    """
    Compute the neighborhood consolidation matrix. 

    adata   
        The AnnData object
    labels 
        Observation labels used to compute the neighborhood
        consolidation matrix. Continuous or discrete labels are allowed, 
        and typically, integer labels are assumed to be discrete.
    compute_similarity
        Whether to return the label similarity matrix. Only useful if 
        discrete labels are provided. 
    key_added
        If not specified, the neighborhood consolidation matrix is stored as
        :attr:`~anndata.AnnData.obsm`\\ `['X_nbc']`, and the parameters as 
        :attr:`~anndata.AnnData.uns`\\ `['nbc_params']`. 
        The index is assumed to be stored as 
        :attr:`~anndata.AnnData.obsm`\\`['index']`

        If specified, ``[key_added]`` is prepended to the default keys.
    copy 
        If ``copy=True``, return the neighborhood consolidation matrix instead 
        of updating adata.
    """

    if key_added is None:
        nbc_uns_key = "nbc_params"
        nbc_key = "X_nbc"
        index_key = "index"
    else:
        nbc_key = key_added + "_nbc"
        nbc_uns_key = key_added + "_nbc_params"
        index_key = key_added + "_index"

    if index_key in adata.obsm.keys():
        Index = adata.obsm[index_key]
    else:
        raise ValueError("Must run ``concordex.neighbors.compute_neighbors()``")

    labels = Labels(labels)
    labels.extract(adata)

    if labels.labeltype != "discrete" and compute_similarity:
        compute_similarity = False
        warnings.warn("Expected discrete labels to compute similarity matrix")
    

    if compute_similarity:
        print("Computing neighborhood consolidation and similarity matrices...\n")
        nbc, sim = _consolidate(Index, labels, compute_similarity)
    else:
        print("Computing neighborhood consolidation matrix...\n")
        nbc = _consolidate(Index, labels, compute_similarity)

    adata.uns[nbc_uns_key] = {}
    nbc_index_dict = adata.uns[nbc_uns_key]
    
    nbc_index_dict = {
        "nbc_key": nbc_key,
        "labels": labels,
        "labels_found" : labels.labelnames,
        "nbc_colnames" : labels.nbccolumns,
        "params": {
            "compute_similarity": compute_similarity
        },
    }

    if compute_similarity:
        nbc_index_dict['similarity'] = sim
        nbc_index_dict['labelorder'] = labels.discretelabelsunique

    if copy:
        return nbc, nbc_index_dict
    
    # Update adata
    adata.uns[nbc_uns_key] = nbc_index_dict
    adata.obsm[nbc_key] = nbc 
   

def _consolidate(X, labels, compute_similarity):

    def take_col_means(indices, take_from, take_by='row'):
        if take_by == 'row':
            axis=0
        else:
            axis=1
        sub = np.take(take_from, indices, axis=axis)

        return sub.mean(axis=0)

    labels_values = labels.values

    Nbc = np.apply_along_axis(take_col_means, 1, X, take_from=labels_values)

    if compute_similarity:
        nlab = labels.n_unique_labels
        labels_new = labels.discretelabelscollapsed

        labels_uniq = labels.discretelabelsunique
        
        Sim = np.empty((nlab, nlab), dtype=np.float64)
        for i, lab in enumerate(labels_uniq):
            m = np.isin(labels_new, lab)

            Sim[i,:] = Nbc[m, :].mean(axis=0)     

        return Nbc, Sim
    
    return Nbc
   

def compute_neighbors(
    adata: AnnData,
    *,
    use_rep: str | None = None, 
    n_neighbors: int = 30, 
    metric: str = "euclidean", 
    metric_params: dict | None = None,
    n_jobs: int | None = None,
    key_added: str | None = None,
    recompute_index: bool = False, 
    copy: bool = False,
    **kwargs
):
    """
    A very thin wrapper around `sklearn.neighbors.NearestNeighbors`

    adata 
        The adata object
    use_rep 
        Key in adata.obsm to use for constructing the kNN graph
    n_neighbors 
        Number of neighbors used to compute the kNN graph. Defaults to 30.
    metric  
        Metric used to compute distance
    metric_params 
        Additional params passed to metric function
    n_jobs
        Used to control parallel evaluation
    key_added 
        Key which controls where the results are saved if ``copy = False``.
    recompute_index
        If a neighborhood graph exists at the specified key, should the 
        data be overwritten? 
    copy : bool
        If ``copy = True``, return the nearest neighbor graph instead of 
        updating adata.
    **kwargs 
        Additional keyword arguments passed to sklearn.neighbors.NearestNeighbors
    """
    if use_rep is None or use_rep == 'X': 
        X = adata.X
    else: 
        if use_rep in adata.obsm.keys(): 
            X = adata.obsm[use_rep]
        else:
            raise ValueError(
                f"Did not find {use_rep} in ``.obsm.keys()``. "
            )

    nn_kwargs = {}
    if kwargs:
        nn_kwargs = kwargs
    if metric_params is not None: 
        if 'p' in metric_params.keys():
            p = metric_params.pop('p')
            nn_kwargs['p'] = p

        nn_kwargs['metric_params'] = metric_params
    
    if key_added is None:
        index_uns_key = "index_params"
        index_key = "index"
    else:
        index_uns_key = key_added + "_index_params"
        index_key = key_added + "_index"

    index_exists = index_key in adata.obsm.keys() 

    print("Computing nearest neighbors...\n")
    if index_exists and not recompute_index:
        warnings.warn(
            f"A neighborhood graph already exists at ``adata.obsm[{index_key}]``. \
              Set ``recompute_index = TRUE`` to overwrite the existing graph.")
            
        Index = adata.obsm[index_key]
        neighbors_index_dict = adata.uns[index_uns_key]
    
    if recompute_index or not index_exists: 
       
        Index = _compute_neighbors(
            X, n_neighbors=n_neighbors, metric=metric, n_jobs=n_jobs, **nn_kwargs
        )

        adata.uns[index_uns_key] = {}
        neighbors_index_dict = adata.uns[index_uns_key]

        neighbors_index_dict = {
            "index_key": index_key,
            "params": {
                "n_neighbors": n_neighbors,
                "metric": metric,
            }
        }

        if nn_kwargs:
            neighbors_index_dict['params']['nn_kwargs'] = nn_kwargs

        if use_rep is not None:
            neighbors_index_dict["params"]["use_rep"] = use_rep

    if copy:
        return Index  

    # Update adata
    adata.uns[index_uns_key] = neighbors_index_dict
    adata.obsm[index_key] = Index 
    
def _compute_neighbors(
    X, 
    *,
    n_neighbors: int = 30, 
    include_self: bool = False,
    **kwargs
):

    N = X.shape[0]
    
    if include_self:
        index = 0
    else: 
        index = 1
        n_neighbors = n_neighbors+1 
    
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, **kwargs)
    nbrs.fit(X)

    g = nbrs.kneighbors(X, return_distance=False)

    return g[:, index:]

