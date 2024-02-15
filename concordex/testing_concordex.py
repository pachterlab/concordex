import scipy.io
import numpy as np

from concordex_map import concordex_map
from concordex_stat import concordex_stat

knn = scipy.io.mmread('concordex/knn-log-amb.mtx')
labels = np.random.randint(1, 4, size=(knn.getnnz(), 1))
print(labels)

print(concordex_map(knn, labels, 1))