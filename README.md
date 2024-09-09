# concordex 1.0.0

The goal of `concordex` is to identify spatial homogeneous regions (SHRs) as defined in the recent manuscript, [“Identification of spatial homogenous regions in tissues with concordex”](https://doi.org/10.1101/2023.06.28.546949). Briefly, SHRs are are domains that are homogeneous with respect to cell type composition. `concordex` relies on the the k-nearest-neighbor (kNN) graph to representing similarities between cells and uses common clustering algorithms to identify SHRs.

## Installation

`concordex` can be installed via pip
```bash
pip install git+https://github.com/pachterlab/concordex.git
```

## Usage

After installing, `concordex` can be run by simply as follows: 

## Citation

If you’d like to use the `concordex` package in your research, please
cite our recent bioRxiv preprint

> Jackson, K.; Booeshaghi, A. S.; Gálvez-Merchán, Á.; Moses, L.; Chari,
> T.; Pachter, L. Quantitative assessment of single-cell RNA-seq
> clustering with CONCORDEX. bioRxiv (Cold Spring Harbor Laboratory)
> 2023. <https://doi.org/10.1101/2023.06.28.546949>.

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
