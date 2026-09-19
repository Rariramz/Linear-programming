# Optimization Methods in Python

Python and NumPy implementations of classical optimization algorithms, developed for the sixth-semester **Optimization and Control Methods** course. The project covers linear programming, transportation problems, and quadratic programming, with step-by-step traces of intermediate calculations.

## Algorithms

| Method | Scope | Implementation |
| --- | --- | --- |
| [Matrix inverse update](matrix_inversion/README.md) | Update an inverse after replacing one column | [Source](matrix_inversion/core.py) |
| [Primal simplex: main phase](simplex_method/README.md) | Optimize from a feasible starting basis | [Source](simplex_method/core.py) |
| [Simplex: initial phase](initial_stage_simplex_method/README.md) | Construct a feasible basis through an auxiliary problem | [Source](initial_stage_simplex_method/main.py) |
| [Dual simplex](dual_simplex_method/README.md) | Restore primal feasibility from a dual-feasible basis | [Source](dual_simplex_method/main.py) |
| [Transportation optimization](matrix_transport_problem/README.md) | Northwest corner initialization followed by cost improvement | [Source](matrix_transport_problem/main.py) |
| [Quadratic programming](quadratic_programming/README.md) | Optimize a quadratic objective under linear constraints | [Source](quadratic_programming/main.py) |

The implementations expose basis changes, matrix operations, and intermediate solutions for inspection. Each entry point includes a sample problem. Comments and console traces are primarily in Russian.

## Featured case study

[Minimum-cost delivery planning](docs/transportation-case-study.md) models deliveries from three warehouses to three destinations. The optimized allocation reduces cost from **4500 to 3900 monetary units (13.3%)** relative to the northwest corner initialization. The case study includes the formulation, shipment plan, and a mathematical optimality certificate for the coursework instance.

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
python matrix_transport_problem/main.py
```

If PowerShell prevents activation, use `.\.venv\Scripts\python.exe` in place of `python` for installation and execution.

The transportation example allocates shipments from three suppliers to three destinations, satisfying supply and demand while minimizing total shipping cost. It prints the initial allocation and the subsequent optimization trace.

Other examples:

```sh
python matrix_inversion/main.py
python simplex_method/main.py
python initial_stage_simplex_method/main.py
python dual_simplex_method/main.py
python quadratic_programming/main.py
```

Examples use embedded input data and require no interactive input.

## Validation

Run the regression tests after installing the requirements:

```sh
python -m unittest discover -s tests -v
```

Six tests cover the original examples: the matrix inverse identity, initial-phase feasibility, and feasibility plus optimality conditions for the optimization examples. The initial-phase test checks both copies of the implementation, including removal of a redundant constraint. Tests use Python's built-in `unittest` module and require no additional dependency.

## Project status

Each top-level algorithm directory represents a separate university assignment. Reusable implementations remain owned by the task that introduced them: [`matrix_inversion/core.py`](matrix_inversion/core.py) provides the inverse update, and [`simplex_method/core.py`](simplex_method/core.py) provides the main primal-simplex phase. Later assignments import these modules while keeping their own entry points and examples.

This repository preserves the original coursework implementations. It currently consists of standalone scripts with embedded examples and some duplicated helpers. The original examples have mathematical regression checks. Reference-solver comparisons and compatibility beyond the recorded environment have not yet been established; coverage of these examples is not comprehensive algorithm validation.

The initial simplex example demonstrates feasibility search only. Numerical robustness and behavior on degenerate, infeasible, and unbounded inputs require further validation.

## Further documentation

The [learning notes](docs/learning-notes.md) provide a mathematical refresher, a worked transportation example, a glossary of code notation, and a suggested study sequence.

If redirected console output raises an encoding error on Windows, add `-X utf8`, for example: `python -X utf8 matrix_transport_problem/main.py`.
