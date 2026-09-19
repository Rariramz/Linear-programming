# Initial phase of the simplex method

Constructs a feasible starting basis using artificial variables. It can also identify an inconsistent system or remove a redundant constraint.

```sh
python -m initial_stage_simplex_method.main
```

The example contains two dependent constraints and returns the feasible plan `x = (0, 0, 0)`. This task finds a starting plan; it does not optimize the original objective.

The method is in [`core.py`](core.py); [`main.py`](main.py) contains the example.
