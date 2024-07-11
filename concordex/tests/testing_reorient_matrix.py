import numpy as np
from scipy.sparse import csr_matrix

def check_matrix_dims(x, k):
    dims = np.array(np.shape(x))
    
    def guess_orientation(x, k, dims):
        if np.diff(dims) == 0:
            if np.all(np.sum(x, axis=1) / k) == 1:
                return 1
            if np.all(np.sum(x, axis=0) / k) == 1:
                return 2
        else:
            axis = np.where(dims == k)[0] # axis = which(dims == k)
            if len(axis) == 0:
                return None
            if axis[0] == 0:
                return 3
            if axis[0] == 1:
                return 4

        return None
    
    pattern = guess_orientation(x, k=k, dims=dims)

    if pattern is None:
        raise ValueError("Cannot determine whether neighbors are oriented on the rows or columns")

    return dims, {
        1: "none",
        2: "transpose",
        3: "expand_row",
        4: "expand_col"
    }[pattern]

def reorient_matrix(x, k, how):
    dims, _ = check_matrix_dims(x, k) # look closely at this
    r, c = dims

    if how == "none":
        return x
    elif how == "transpose":
        return np.transpose(x)
    elif how == "expand_row":
        i = np.sort(np.repeat(np.arange(c), k))
        j = np.ravel(x)
        data = np.ones(c * k)
        return csr_matrix((data, (i, j)), shape=(c, c))
    elif how == "expand_col":
        i = np.repeat(np.arange(r), k)
        j = np.ravel(x)
        data = np.ones(r * k)
        return csr_matrix((data, (i, j)), shape=(r, r))
    else:
        raise ValueError("Invalid 'how' parameter")
    
matrix_expand_row = np.array([[0, 1],
                              [2, 1],
                              [1, 0]])

k_expand_row = 2
reoriented_expand_row = reorient_matrix(matrix_expand_row, k_expand_row, "expand_col")
print(reoriented_expand_row)  # Should return a sparse matrix with expanded rows