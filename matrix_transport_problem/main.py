import numpy as np

from .core import matrix_transport_problem


if __name__ == '__main__':
    supply = np.array([100, 300, 300])
    demand = np.array([300, 200, 200])
    costs = np.array([[8, 4, 1],
                      [8, 4, 3],
                      [9, 7, 5]])

    matrix_transport_problem(supply, demand, costs)
