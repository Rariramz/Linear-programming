import numpy as np


def north_west_corner_rule(vector_a, vector_b):
    m, n = len(vector_a), len(vector_b)
    matrix_x = np.zeros([m, n])
    array_b = []
    i, j = 0, 0

    a = vector_a.copy()
    b = vector_b.copy()
    while i < m and j < n:
        array_b.append((i, j))
        if a[i] < b[j]:
            matrix_x[i][j] = a[i]
            b[j] -= a[i]
            i += 1
        else:
            matrix_x[i][j] = b[j]
            a[i] -= b[j]
            j += 1

    return matrix_x, array_b
