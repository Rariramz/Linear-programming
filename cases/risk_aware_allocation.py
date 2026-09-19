import numpy as np

from quadratic_programming import quadratic_programming_task


def solve():
    linear_cost = np.array([-5., -6., -4.])
    risk = np.diag([0.1, 0.2, 0.05])
    budget = np.array([[1., 1., 1.]])
    initial_plan = np.array([100., 0., 0.])
    plan, _ = quadratic_programming_task(
        linear_cost, risk, budget, initial_plan,
        np.array([1]), np.array([1]), logger=None)
    objective = linear_cost @ plan + 0.5 * plan @ risk @ plan
    return plan, float(objective)


if __name__ == '__main__':
    plan, objective = solve()
    print(f'Allocation: {np.round(plan, 2)}')
    print(f'Objective value: {objective:.2f}')
