import numpy as np
from anndata import AnnData

from neighbors import (
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
    metric  
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

    # 2. Then consolidatescm-history-item:/Users/kaylajackson/Library/CloudStorage/OneDrive-CaliforniaInstituteofTechnology/Projects/concordex?%7B%22repositoryId%22%3A%22scm0%22%2C%22historyItemId%22%3A%2275c203ffcfea81cbeb3d4e7a74927c49d90ee3c7%22%2C%22historyItemParentId%22%3A%2224721cfd2af304a651af28d6d6c140e74e1c14d4%22%2C%22historyItemDisplayId%22%3A%2275c203f%22%7D
    consolidate(adata, labels, key_added=key_added, compute_similarity=compute_similarity)


