# Optimization Methods in Python

Python and NumPy implementations of classical optimization algorithms, developed for the sixth-semester **Optimization and Control Methods** course. The project covers linear programming, transportation problems, and quadratic programming, with step-by-step traces of intermediate calculations.

## Algorithms

| Method | Scope | Implementation |
| --- | --- | --- |
| Matrix inverse update | Update an inverse after replacing one column | [Source](matrix_inversion/main.py) |
| Primal simplex: main phase | Optimize from a feasible starting basis | [Source](simplex_method/main.py) |
| Simplex: initial phase | Construct a feasible basis through an auxiliary problem | [Source](initial_stage_simplex_method/main.py) |
| Dual simplex | Restore primal feasibility from a dual-feasible basis | [Source](dual_simplex_method/main.py) |
| Transportation optimization | Northwest corner initialization followed by cost improvement | [Source](matrix_transport_problem/main.py) |
| Quadratic programming | Optimize a quadratic objective under linear constraints | [Source](quadraic_programming/main.py) |

The implementations expose basis changes, matrix operations, and intermediate solutions for inspection. Each entry point includes a sample problem. Comments and console traces are primarily in Russian.

## Featured case study

[Minimum-cost delivery planning](docs/transportation-case-study.md) models deliveries from three warehouses to three destinations. The optimized allocation reduces cost from **4500 to 3900 monetary units (13.3%)** relative to the northwest corner initialization. The case study includes the formulation, shipment plan, and a mathematical optimality certificate for the coursework instance.

## Quick start

Requires **Python 3** and **NumPy**. From the repository root, create a virtual environment:

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
python -m pip install numpy
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
python quadraic_programming/main.py
```

Run these files directly: imports rely on helper modules in the same directory. Examples use embedded input data and require no interactive input. The path `quadraic_programming` retains the original directory spelling.

## Project status

This repository preserves the original coursework implementations. It currently consists of standalone scripts with embedded examples and some duplicated helpers. There is no automated test suite or reference-solver comparison, and a supported Python/NumPy version range has not yet been established.

The initial simplex example demonstrates feasibility search only. Numerical robustness and behavior on degenerate, infeasible, and unbounded inputs require further validation.

## Further documentation

The [learning notes](docs/learning-notes.md) provide a mathematical refresher, a worked transportation example, a glossary of code notation, and a suggested study sequence.
