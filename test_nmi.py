import numpy as np
from sklearn.metrics.cluster import mutual_info_score, normalized_mutual_info_score


x = np.array([0, 1, 0, 1, 0, 0, 7])
y = np.array([10.9, 30.5, 10.9, 30.5, 10.9, 10.9, 0.1])

# MI: 0.9556998911125343 NMI: 1.0


x = np.array([[0, 1, 2, 3, 0, 3, 2, 1, 0, 2]]).T
y = np.array([[1, 1],
              [1, 0],
              [0, 1],
              [0, 0],
              [1, 1],
              [0, 0],
              [0, 1],
              [1, 0],
              [1, 1],
              [0, 1]])



mi = mutual_info_score(x, y)
nmi = normalized_mutual_info_score(x, y)

print("MI:", mi)
print("NMI:", nmi)