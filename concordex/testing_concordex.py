import scipy.io
import numpy as np

from concordex_map import concordex_map
from concordex_stat import concordex_stat

from scipy.sparse import coo_matrix, csr_matrix 

# sparse = coo_matrix((matrix_none, ))


matrix_none = np.array([[1, 0, 0],
                        [1, 1, 0],
                        [7, 8, 9]])

knn = scipy.io.mmread('concordex/knn-log-amb.mtx')
#print(knn)
# print(np.shape(knn))

labels = np.random.randint(1, 3, size=(knn.getnnz(), 1))
#print(labels)

print(concordex_map(knn, labels, 1))

"""
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

print(guess_orientation(knn, 30, (56528,56528)))"""