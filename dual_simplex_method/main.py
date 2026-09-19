import numpy as np

from .core import dual_simplex_method


if __name__ == '__main__':
    a = np.array([[-2, -1, -4, 1, 0],
                  [-2, -2, -2, 0, 1]])
    b = [-1, -3 / 2]
    c = [-4, -3, -7, 0, 0]
    basis = [3, 4]

    result = dual_simplex_method(2, 5, a, b, c, basis)
    if not result:
        print('\tНет решений.')
