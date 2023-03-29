import math
import numpy as np

m = 16
n = 20
r = 5
lambd = 1
k = 0

while(lambd <= n):
  A = np.zeros((m, n))
  tau = min(lambd + r - 1, n)
  k += 1
  print(f"{k} triangularization")
  A[lambd-1:m, lambd-1:tau] = k
  print(A)
  lambd = tau + 1
