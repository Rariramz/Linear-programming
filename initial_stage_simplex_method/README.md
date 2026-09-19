# Initial phase of the simplex method

Constructs a feasible starting basis using artificial variables. It can also identify an inconsistent system or remove a redundant constraint.

```sh
python initial_stage_simplex_method/main.py
```

The example contains two dependent constraints and returns the feasible plan `x = (0, 0, 0)`. This task finds a starting plan; it does not optimize the original objective.
