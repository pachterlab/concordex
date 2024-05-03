
# Compatibility between ConcordexR and Concordex Python

## Comparison between R and Python functionality

| Output | R Function | Python Script| Python Command Line |
|--------|------------|--------------|---------------------|
| Return Uncorrected Concordex | `res <- calculateConcordex(x, labels, k = 20, n.iter = 15, return.map = FALSE, BPPARAM = SerialParam())`, `$concordex` | `trace, _, _ = calculate_concordex(knn, labels)` | `concordex map [path to knn] -a [path to labels] -o [path to output]` |
| Return Concordex Ratio | `res <- calculateConcordex(x, labels, k = 20, n.iter = 15, return.map = FALSE, BPPARAM = SerialParam())`, `$concordex_ratio` | `_, corrected_trace, _ = calculate_concordex(knn, labels)`   | `concordex stat [path to knn] -a [path to labels] -o [path to output]` | 
| Return Mean Random Concordex | `res <- calculateConcordex(x, labels, k = 20, n.iter = 15, return.map = FALSE, BPPARAM = SerialParam())`, `$mean_random_concordex` | `_, _, random_trace = calculate_concordex(knn, labels)`   | `concordex stat [path to knn] -a [path to labels] -o [path to output]` | 
| Return Concordex Map Matrix | `res <- calculateConcordex(x, labels, k = 20, n.iter = 15, return.map = TRUE, BPPARAM = SerialParam())`, `$map` | `map_mtx = concordex_map(x, labels, k)` | `concordex map [path to knn] -a [path to labels] -o [path to output]` |
| Plot Concordex Map Matrix as Heatmap | `res <- calculateConcordex(x, labels, k = 20, n.iter = 15, return.map = TRUE, BPPARAM = SerialParam())`, `heatConcordex(res)` | N/A | N/A |

## Terminology

- In the Python version, *concordex* is called *trace* or *uncorrected concordex*.
- *Concordex ratio* can be called *corrected trace* or *corrected concordex*.
- *Mean random concordex* and *random_trace* are interchangeable.
- `knn` and `x` both refer to the knn matrix to be inputted (which cannot be a sparse matrix).

rename concordex stat to calculate concordex