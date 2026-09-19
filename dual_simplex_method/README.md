# Dual simplex method

Starts with a dual-feasible basis and restores feasibility of the primal plan while keeping the dual conditions valid.

```sh
python -m dual_simplex_method.main
```

The built-in example returns `x = (0.25, 0.5, 0, 0, 0)` with objective value `-2.5`.

The method is in [`core.py`](core.py); [`main.py`](main.py) contains the example.
