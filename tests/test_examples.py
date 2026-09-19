"""Mathematical regression checks for the original coursework examples."""

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
from numpy.testing import assert_allclose


ROOT = Path(__file__).resolve().parents[1]
ATOL = 1e-9


def load_script(relative_path):
    """Load a standalone script without mixing its local helpers with others."""
    path = ROOT / relative_path
    helper_names = {file.stem for file in path.parent.glob('*.py')}
    saved = {name: sys.modules.pop(name) for name in helper_names if name in sys.modules}
    saved_path = sys.path.copy()
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location('coursework_example', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = saved_path
        for name in helper_names:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


class ExampleTests(unittest.TestCase):
    def assert_feasible(self, a, b, x):
        self.assertTrue(np.all(np.isfinite(x)))
        self.assertTrue(np.all(np.asarray(x) >= -ATOL))
        assert_allclose(a @ x, b, atol=ATOL, rtol=0)

    def assert_lp_optimal(self, a, b, c, x, basis, expected_value):
        self.assert_feasible(a, b, x)
        # A dual-feasible vector with the same objective certifies a maximum.
        u = np.linalg.solve(a[:, basis].T, c[basis])
        self.assertTrue(np.all(a.T @ u - c >= -ATOL))
        self.assertAlmostEqual(float(c @ x), expected_value)
        self.assertAlmostEqual(float(b @ u), expected_value)

    def test_inverse_update(self):
        paths = ('matrix_inversion/core.py', 'matrix_inversion/main.py')
        for path in paths:
            with self.subTest(path=path):
                module = load_script(path)
                a = np.array([[1, -1, 0], [0, 1, 0], [0, 0, 1]])
                inverse = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
                replacement = [1, 0, 1]
                result = module.matrix_inversion(3, 2, inverse, replacement, logger=None)
                a[:, 2] = replacement
                assert_allclose(a @ result, np.eye(3), atol=ATOL, rtol=0)
                assert_allclose(result @ a, np.eye(3), atol=ATOL, rtol=0)

    def test_primal_simplex(self):
        a = np.array([[-1, 1, 1, 0, 0], [1, 0, 0, 1, 0], [0, 1, 0, 0, 1]])
        b = np.array([1, 3, 2])
        c = np.array([1, 1, 0, 0, 0])
        paths = ('simplex_method/core.py', 'simplex_method/main.py')
        for path in paths:
            with self.subTest(path=path):
                module = load_script(path)
                x, basis = module.main_stage_simplex_method(
                    3, 5, a, [0, 0, 1, 3, 2], c, [2, 3, 4], logger=None)
                self.assert_lp_optimal(a, b, c, x, basis, 5)

    def test_initial_simplex_with_redundant_constraint(self):
        # Both copies contain the same recursive basis-cleanup path.
        for path in ('initial_stage_simplex_method/main.py',
                     'dual_simplex_method/initial_stage_simplex_method.py'):
            with self.subTest(path=path):
                module = load_script(path)
                a = np.array([[1, 1, 1], [2, 2, 2]])
                result = module.initial_stage_simplex_method(a.copy(), [0, 0], 2, 3, logger=None)
                self.assertIsInstance(result, tuple)
                x, basis = result
                self.assert_feasible(a, np.zeros(2), x)
                self.assertEqual(len(basis), np.linalg.matrix_rank(a))
                self.assertTrue(all(0 <= j < 3 for j in basis))
                self.assertEqual(np.linalg.matrix_rank(a[:, basis]), len(basis))

    def test_dual_simplex(self):
        module = load_script('dual_simplex_method/main.py')
        a = np.array([[-2, -1, -4, 1, 0], [-2, -2, -2, 0, 1]])
        b = np.array([-1, -1.5])
        c = np.array([-4, -3, -7, 0, 0])
        x, basis = module.dual_simplex_method(2, 5, a, b, c, [3, 4], logger=None)
        self.assert_lp_optimal(a, b, c, x, basis, -2.5)

    def test_transportation(self):
        module = load_script('matrix_transport_problem/main.py')
        supply = np.array([100, 300, 300])
        demand = np.array([300, 200, 200])
        costs = np.array([[8, 4, 1], [8, 4, 3], [9, 7, 5]])
        x, _ = module.matrix_transport_problem(supply.copy(), demand.copy(), costs, logger=None)
        self.assertTrue(np.all(np.isfinite(x)))
        self.assertTrue(np.all(x >= -ATOL))
        assert_allclose(x.sum(axis=1), supply, atol=ATOL, rtol=0)
        assert_allclose(x.sum(axis=0), demand, atol=ATOL, rtol=0)
        self.assertAlmostEqual(float(np.sum(x * costs)), 3900)
        u, v = np.array([0, 2, 4]), np.array([5, 2, 1])
        self.assertTrue(np.all(u[:, None] + v <= costs))
        self.assertAlmostEqual(float(u @ supply + v @ demand), float(np.sum(x * costs)))

    def test_quadratic_programming(self):
        module = load_script('quadraic_programming/main.py')
        c = np.array([-8, -6, -4, -6])
        d = np.array([[2, 1, 1, 0], [1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0]])
        a = np.array([[1, 0, 2, 1], [0, 1, -1, 2]])
        x, _ = module.quadratic_programming_task(
            c, d, a, np.array([2, 3, 0, 0]), np.array([1, 2]), np.array([1, 2]), logger=None)
        self.assert_feasible(a, np.array([2, 3]), x)
        # Convexity plus stationarity, nonnegative reduced gradients, and
        # complementarity certify optimality for this specific instance.
        self.assertTrue(np.all(np.linalg.eigvalsh(d) >= -ATOL))
        reduced_gradient = c + d @ x + a.T @ np.array([2.2, 1.9])
        self.assertTrue(np.all(reduced_gradient >= -ATOL))
        assert_allclose(x * reduced_gradient, np.zeros(4), atol=ATOL, rtol=0)
        self.assertAlmostEqual(float(c @ x + 0.5 * x @ d @ x), -19.95)


if __name__ == '__main__':
    unittest.main()
