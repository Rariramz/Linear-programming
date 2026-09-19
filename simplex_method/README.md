# Main phase of the simplex method

Finds an optimal plan from a supplied feasible basic plan. Each iteration calculates reduced costs, selects an entering variable, applies the ratio test, and updates the basis.

```sh
python -m simplex_method.main
```

The built-in example returns `x = (3, 2, 2, 0, 0)` with objective value `5`. The reusable implementation is in [`core.py`](core.py).
