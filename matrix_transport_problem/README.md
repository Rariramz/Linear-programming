# Transportation problem

Builds an initial shipping plan with the northwest corner rule and improves it using the potential method.

```sh
python -m matrix_transport_problem.main
```

For the built-in example, the initial cost is `4500` and the final cost is `3900`. The full calculation is described in the [case study](../docs/transportation-case-study.md).

The main method is in [`core.py`](core.py); [`main.py`](main.py) contains the example.
