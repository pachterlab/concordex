from .concordex_map import check_matrix_dims, reorient_matrix

import numpy as np
from scipy.sparse import csr_matrix

# Testing for no orientation
matrix_none = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])

k_none = 3
print(check_matrix_dims(matrix_none, k_none))  # Should return 'none'
reoriented_none = reorient_matrix(matrix_none, k_none, "none")
print(reoriented_none)  # Should return the same matrix

# Testing for transpose orientation
matrix_transpose = np.array([[1, 2, 3],
                             [4, 5, 6]])

k_transpose = 2
print(check_matrix_dims(matrix_transpose, k_transpose))  # Should return 'transpose'
reoriented_transpose = reorient_matrix(matrix_transpose, k_transpose, "transpose")
print(reoriented_transpose) # Return transposed matrix

# Testing for expand_row orientation
matrix_expand_row = np.array([[1, 2],
                              [3, 4],
                              [5, 6]])

k_expand_row = 2
print(check_matrix_dims(matrix_expand_row, k_expand_row))  # Should return 'expand_row'
reoriented_expand_row = reorient_matrix(matrix_expand_row, k_expand_row, "expand_row")
print(reoriented_expand_row)  # Should return a sparse matrix with expanded rows

# Testing for expand_col orientation
matrix_expand_col = np.array([[1, 2, 3],
                              [4, 5, 6]])

k_expand_col = 2
print(check_matrix_dims(matrix_expand_col, k_expand_col))  # Should return 'expand_col'
reoriented_expand_col = reorient_matrix(matrix_expand_col, k_expand_col, "expand_col")
print(reoriented_expand_col)  # Should return a sparse matrix with expanded columns
