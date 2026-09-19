import numpy as np

from simplex_method import main_stage_simplex_method


def solve():
    constraints = np.array([[2., 1., 1., 0., 0.],
                            [1., 1.5, 0., 1., 0.],
                            [1., 0., 0., 0., 1.]])
    profit = np.array([40., 30., 0., 0., 0.])
    initial_plan = np.array([0., 0., 100., 90., 40.])
    plan, basis = main_stage_simplex_method(
        3, 5, constraints, initial_plan, profit, [2, 3, 4], logger=None)
    return plan, float(profit @ plan), basis


if __name__ == '__main__':
    plan, profit, _ = solve()
    print(f'Product A: {plan[0]:.0f}')
    print(f'Product B: {plan[1]:.0f}')
    print(f'Maximum profit: {profit:.0f}')
