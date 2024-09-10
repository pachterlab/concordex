import numpy as np
from anndata import AnnData

from ..neighbors._neighborhood import (
    compute_neighbors, 
    consolidate
    )


def calculate_concordex(
    adata: AnnData, 
    labels, 
    *,
    n_neighbors: int = 30,
    use_rep: str | None = "X", 
    metric: str = "euclidean", 
    metric_params: dict | None = None,
    n_jobs: int | None = None,
    index_key: str = "index",
    compute_similarity: bool = False
):
    """
    adata   
        The AnnData object
    labels 
        Observation labels used to compute the neighborhood
        consolidation matrix. Continuous or discrete labels are allowed, 
        and typically, integer labels are assumed to be discrete.
    n_neighbors 
        Number of neighbors used to compute the kNN graph. Defaults to 30.
    metric  
    metric_params 
        Additional parameters passed to metric function
    n_jobs
        Used to control parallel evaluation
    index_key
        The key in `adata.obsm` storing the neighborhood information. 
        Usually from running concordex.neighbors.compute_neighbors(), otherwise, 
        this should point to an nxk matrix, where the entries in each column are the
        indices of the k nearest-neighbors of the given observation. 
    compute_similarity
        Whether to return the label similarity matrix and stores this information in 
        adata.uns['nbc_params']['similarity']. Only useful if discrete labels are 
        provided. 

    """

    if index_key and index_key in adata.obsm.keys():
        consolidate(adata, labels, 
            index_key=index_key, compute_similarity=compute_similarity)
        
    else: 
        if index_key is None:
            index_key = "index"

        compute_neighbors(adata, 
            use_rep=use_rep, n_neighbors=n_neighbors, metric=metric, 
            metric_params=metric_params, n_jobs=n_jobs)

        consolidate(adata, labels, 
            index_key=index_key, compute_similarity=compute_similarity)


