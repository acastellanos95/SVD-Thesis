import math
import numpy as np

n = 5

A = np.zeros((n, n))

for i in range(n):
    A[i, i] = np.random.rand()
    if i > 0:
        A[i - 1, i] = np.random.rand()

print(A)

B = np.matmul(np.transpose(A), A)

print(B)
