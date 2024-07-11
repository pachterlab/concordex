import numpy as np

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
            if 0 in axis:
                return 3
            if 1 in axis:
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

def guess_orientation(x, k, dims):
    if np.diff(dims) == 0:
        if np.all(np.sum(x, axis=1) / k) == 1:
            return 1
        if np.all(np.sum(x, axis=0) / k) == 1:
            return 2
    else:
        axis = np.where(dims == k)# axis = which(dims == k)
        if len(axis) == 0:
            return None
        if 0 in axis:
            return 3
        if 1 in axis:
            return 4

    return None

mtx = np.array([[1, 2, 3],[4, 5, 6]])

k = 2
dims = np.array(np.shape(mtx))
print(guess_orientation(mtx, k, dims))

print(np.where(dims == 2)[0])