import numpy as np

from .core import initial_stage_simplex_method


if __name__ == '__main__':
    a = np.array([[1, 1, 1],
                  [2, 2, 2]])
    b = [0, 0]

    initial_stage_simplex_method(a, b, 2, 3)
