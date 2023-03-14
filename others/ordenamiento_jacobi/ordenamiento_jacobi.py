import math
import numpy as np

n = 10
m = math.floor((n + 1) / 2)
print(m)

A = np.zeros((n, n))

for k in range(1, m):
    for q in range(m - k + 1, n - k + 1):
        if m - k + 1 <= q <= 2 * m - 2 * k:
            p = (2 * m - 2 * k + 1) - q
        elif 2 * m - 2 * k < q <= 2 * m - k - 1:
            p = (4 * m - 2 * k) - q
        elif 2 * m - k - 1 < q:
            p = n
        print(f"({p-1},{q-1})")
        A[p - 1, q - 1] = k

for k in range(m, 2 * m):
    for q in range(4 * m - n - k, 3 * m - k):
        if q < 2 * m - k + 1:
            p = n
        elif 2 * m - k + 1 <= q <= 4 * m - 2 * k - 1:
            p = (4 * m - 2 * k) - q
        elif 4 * m - 2 * k - 1 < q:
            p = (6 * m - 2 * k - 1) - q
        print(f"({p-1},{q-1})")
        A[p - 1, q - 1] = k

print(A)

# A = np.zeros((n, n))
#
# for k in range(1, n):
#     if k > 2:
#         for i in range(n - k + 2, n - math.floor(k / 2) + 1):
#             j = 2 * n - k + 2 - i
#             A[i - 1, j - 1] = k
#     else:
#         for i in range(1, math.ceil((n - k) / 2) + 1):
#             j = (n - k + 2) - i
#             A[i - 1, j - 1] = k
#
# for i in range(2, math.ceil(n / 2) + 1):
#     j = (n + 2) - i
#     A[i - 1, j - 1] = n
#
# print(A)