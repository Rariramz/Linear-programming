import numpy as np

from .core import quadratic_programming_task


if __name__ == '__main__':
    c = np.array([-8, -6, -4, -6])
    d = np.array([[2, 1, 1, 0],
                  [1, 1, 0, 0],
                  [1, 0, 1, 0],
                  [0, 0, 0, 0]])
    a = np.array([[1, 0, 2, 1],
                  [0, 1, -1, 2]])
    x = np.array([2, 3, 0, 0])
    basis = np.array([1, 2])

    quadratic_programming_task(c, d, a, x, basis, basis.copy())
