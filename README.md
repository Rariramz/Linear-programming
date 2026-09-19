# Optimization Methods in Python

Python and NumPy implementations of classical optimization algorithms, developed for the sixth-semester **Optimization and Control Methods** course. The project covers linear programming, transportation problems, and quadratic programming, with step-by-step traces of intermediate calculations.

## Algorithms

| Method                                                            | Scope                                                        | Implementation                                 |
| ----------------------------------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------- |
| [Matrix inverse update](matrix_inversion/README.md)               | Update an inverse after replacing one column                 | [Source](matrix_inversion/core.py)             |
| [Primal simplex: main phase](simplex_method/README.md)            | Optimize from a feasible starting basis                      | [Source](simplex_method/core.py)               |
| [Simplex: initial phase](initial_stage_simplex_method/README.md)  | Construct a feasible basis through an auxiliary problem      | [Source](initial_stage_simplex_method/core.py) |
| [Dual simplex](dual_simplex_method/README.md)                     | Restore primal feasibility from a dual-feasible basis        | [Source](dual_simplex_method/core.py)          |
| [Transportation optimization](matrix_transport_problem/README.md) | Northwest corner initialization followed by cost improvement | [Source](matrix_transport_problem/core.py)     |
| [Quadratic programming](quadratic_programming/README.md)          | Optimize a quadratic objective under linear constraints      | [Source](quadratic_programming/core.py)        |

The implementations expose basis changes, matrix operations, and intermediate solutions for inspection. Each entry point includes a sample problem. Comments and console traces are primarily in Russian.

## Case studies

- [Production planning](docs/production-planning-case-study.md) maximizes profit for two products under three capacity constraints. The simplex solution earns **2400**.
- [Minimum-cost delivery planning](docs/transportation-case-study.md) allocates shipments from three warehouses to three destinations. Cost falls from **4500 to 3900 (13.3%)**.
- [Risk-aware allocation](docs/risk-aware-allocation-case-study.md) distributes a fixed budget using a quadratic benefit-risk objective.

All three use synthetic data. Each write-up includes the formulation, result, and a mathematical optimality check.

## Quick start

Requires **Python 3** and **NumPy**. All six built-in examples were run successfully with **Python 3.14.3 and NumPy 2.5.3 on Windows**; NumPy is pinned in `requirements.txt`. From the repository root, create a virtual environment:

```sh
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```sh
source .venv/bin/activate
```

Install NumPy and run the transportation example:

```sh
python -m pip install -r requirements.txt
python -m matrix_transport_problem.main
```

If PowerShell prevents activation, use `.\.venv\Scripts\python.exe` in place of `python` for installation and execution.

The transportation example allocates shipments from three suppliers to three destinations, satisfying supply and demand while minimizing total shipping cost. It prints the initial allocation and the subsequent optimization trace.

Other examples:

```sh
python -m matrix_inversion.main
python -m simplex_method.main
python -m initial_stage_simplex_method.main
python -m dual_simplex_method.main
python -m quadratic_programming.main
```

The additional applied examples run with:

```sh
python -m cases.production_planning
python -m cases.risk_aware_allocation
```

Examples use embedded input data and require no interactive input.

## Validation

Run the regression tests after installing the requirements:

```sh
python -m unittest discover -s tests -v
```

Eight tests cover the six coursework examples and the two additional applied cases. They check feasibility, objective values, matrix identities, and optimality conditions. Tests use Python's built-in `unittest` module and require no additional dependency.

## Project status

Each top-level algorithm directory represents a separate university assignment. Reusable implementations remain owned by the task that introduced them: [`matrix_inversion/core.py`](matrix_inversion/core.py) provides the inverse update, and [`simplex_method/core.py`](simplex_method/core.py) provides the main primal-simplex phase. Later assignments import these modules while keeping their own entry points and examples.

This repository preserves the original coursework methods while exposing them as reusable Python modules. The documented examples have mathematical regression checks. Reference-solver comparisons and compatibility beyond the recorded environment have not yet been established; coverage of these examples is not comprehensive algorithm validation.

The initial simplex example demonstrates feasibility search only. Numerical robustness and behavior on degenerate, infeasible, and unbounded inputs require further validation.

## Further documentation

The [learning notes](docs/learning-notes.md) provide a mathematical refresher, a glossary of code notation, and a suggested study sequence.

If redirected console output raises an encoding error on Windows, add `-X utf8`, for example: `python -X utf8 -m matrix_transport_problem.main`.
