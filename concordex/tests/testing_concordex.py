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
# print(np.shape(knn))

labels = np.random.randint(1, 3, size=(knn.getnnz(), 1))

print(concordex_map(knn, labels, 1))