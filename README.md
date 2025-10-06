# concordex 1.1.1

The goal of `concordex` is to identify spatial homogeneous regions (SHRs) as defined in the recent manuscript, [“Identification of spatial homogenous regions in tissues with concordex”](https://doi.org/10.1101/2023.06.28.546949). Briefly, SHRs are are domains that are homogeneous with respect to cell type composition. `concordex` relies on the the k-nearest-neighbor (kNN) graph to representing similarities between cells and uses common clustering algorithms to identify SHRs.

## Installation

`concordex` can be installed via pip
```bash 
pip install concordex
```
.... and from Github 
```bash
pip install git+https://github.com/pachterlab/concordex.git
```

## Usage

After installing, `concordex` can be run as follows: 
```
import scanpy as sc 
from concordex.tools import calculate_concordex

ad = sc.datasets.pbmc68k_reduced()

# Compute concordex with discrete labels
calculate_concordex(ad, 'louvain', n_neighbors=10)

# Neighborhood consolidation information is stored in `adata.obsm`
ad.obsm['X_nbc'][:3]

# The column names are stored in `adata.uns`
ad.uns['nbc_params']['nbc_colnames']
```

## Citation

If you’d like to use the `concordex` package in your research, please
cite our recent bioRxiv preprint:

> Jackson, K.; Booeshaghi, A. S.; Gálvez-Merchán, Á.; Moses, L.; Chari,
> T.; Kim, A.; Pachter, L. Identification of spatial homogeneous regions in tissues 
> with concordex. bioRxiv (Cold Spring Harbor Laboratory) 2023. 
> <https://doi.org/10.1101/2023.06.28.546949>.

    @article {Jackson2023.06.28.546949, 
        author = {Jackson, Kayla C. and Booeshaghi, A. Sina and G{'a}lvez-Merch{'a}n, {'A}ngel and Moses, Lambda and Chari, Tara and Kim, Alexandra and Pachter, Lior}, 
        title = {Identification of spatial homogeneous regions in tissues with concordex}, 
        year = {2024}, 
        doi = {10.1101/2023.06.28.546949}, 
        publisher = {Cold Spring Harbor Laboratory}, 
        URL = {<https://www.biorxiv.org/content/early/2024/07/18/2023.06.28.546949>},
        journal = {bioRxiv} 
    }

## Maintainer

[Kayla Jackson](https://github.com/kayla-jackson)
