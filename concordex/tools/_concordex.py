# import numpy as np
from anndata import AnnData

from ..neighbors import (
    compute_neighbors, 
    consolidate
    )


def calculate_concordex(
    adata: AnnData, 
    labels, 
    *,
    n_neighbors: int = 30,
    use_rep: str | None = "spatial", 
    metric: str = "euclidean", 
    metric_params: dict | None = None,
    n_jobs: int | None = None,
    key_added: str | None = None,
    compute_similarity: bool = False, 
    recompute_neighbors: bool = False
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
    metric  from
        Metric used to compute distance
    metric_params 
        Additional parameters passed to metric function
    n_jobs
        Used to control parallel evaluation
    key_added
        If not specified, the relevant results are stored as follows:
        The neighborhood information is stored as, 
        :attr:`~anndata.AnnData.obsm`\\ `['adjacency']`, the neighborhood consolidation matrix as 
        :attr:`~anndata.AnnData.obsm`\\ `['nbc']`, and the parameters as 
        :attr:`~anndata.AnnData.uns`\\ `['adj_params']` and 
        :attr:`~anndata.AnnData.uns`\\ `['nbc_params']`.
        If specified, ``[key_added]`` is prepended to the default keys.
    compute_similarity
        Whether to return the label similarity matrix and stores this information in 
        adata.uns['nbc_params']['similarity']. Only implemented for discrete labels. 
    recompute_neighbors
        If a neighborhood graph exists at the specified key, should the 
        data be overwritten? 
    """
    
    # 1. Compute neighborhood graph
    compute_neighbors(adata, 
        use_rep=use_rep, n_neighbors=n_neighbors, metric=metric, 
        metric_params=metric_params, n_jobs=n_jobs, key_added=key_added, recompute_neighbors=recompute_neighbors)

    # 2. Then consolidate
    consolidate(adata, labels, key_added=key_added, compute_similarity=compute_similarity)


