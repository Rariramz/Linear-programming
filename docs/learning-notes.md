# Learning Notes: Optimization Methods

A refresher for returning to this coursework after time away. These notes explain the mathematics, connect it to the existing scripts, and suggest a path through the project. For the public project overview and installation instructions, see the [README](../README.md).

## Contents

- [A refresher: what is optimization?](#a-refresher-what-is-optimization)
- [Implemented methods](#implemented-methods)
- [Getting started](#getting-started)
- [Worked example: planning deliveries](#worked-example-planning-deliveries)
- [Understanding simplex and the code notation](#understanding-simplex-and-the-code-notation)
- [Suggested learning order](#suggested-learning-order)
- [Current limitations and next steps](#current-limitations-and-next-steps)

## A refresher: what is optimization?

Optimization means choosing the best available decision subject to restrictions. For example, a factory might choose how many units of each product to make to maximize profit, while respecting limits on materials and working hours.

An optimization model has three ingredients:

| Ingredient         | Meaning                                  | Factory example                            |
| ------------------ | ---------------------------------------- | ------------------------------------------ |
| Decision variables | Quantities we can choose                 | Units of each product to manufacture       |
| Objective function | The quantity we maximize or minimize     | Total profit                               |
| Constraints        | Rules that a valid decision must satisfy | Material use cannot exceed available stock |

A **feasible solution** satisfies every constraint. An **optimal solution** is a feasible solution with the best objective value. There can be multiple optimal solutions.

In **linear programming**, the objective and constraints are linear: variables are multiplied by fixed coefficients and added together. There are no squared variables or products between decision variables. Here, “programming” means mathematical planning.

A standard equality form for a maximization problem is:

$$
\max_x c^T x \qquad \text{subject to } Ax = b,\quad x \geq 0.
$$

- $x$ is the vector of decisions.
- $c$ contains the objective coefficients, such as profit per unit.
- $A$ describes how decisions contribute to each constraint.
- $b$ contains the required totals or resource limits after conversion to equality form.
- $c^T x$ is the weighted sum of the decisions; the superscript $T$ denotes transpose.

An inequality can be converted into an equality using a **slack variable**. For example, $2x_1 + x_2 \leq 10$ becomes $2x_1 + x_2 + s = 10$, with $s \geq 0$. The slack records unused capacity.

Two outcomes should be distinguished from an optimum:

- **Infeasible:** no decision satisfies all constraints.
- **Unbounded:** feasible decisions exist, and the objective can keep improving without a finite limit.

Linear programming normally permits fractional decisions. Requiring whole numbers leads to integer programming, which is outside the scope of this repository.

## Implemented methods

| Directory                                                                  | Method                                    | What it is intended to do                                                                          |
| -------------------------------------------------------------------------- | ----------------------------------------- | -------------------------------------------------------------------------------------------------- |
| [`matrix_inversion/`](../matrix_inversion/core.py)                         | Inverse update after replacing one column | Update a known matrix inverse, a calculation used when a simplex basis changes                     |
| [`simplex_method/`](../simplex_method/core.py)                             | Main phase of the simplex method          | Improve an existing feasible basic solution toward an optimum                                      |
| [`initial_stage_simplex_method/`](../initial_stage_simplex_method/core.py) | Initial phase of the simplex method       | Find a feasible starting basis using an auxiliary problem, or detect infeasibility                 |
| [`dual_simplex_method/`](../dual_simplex_method/core.py)                   | Dual simplex method                       | Start from a dual-feasible basis and repair primal infeasibility while preserving dual feasibility |
| [`matrix_transport_problem/`](../matrix_transport_problem/core.py)         | Transportation optimization               | Construct an initial shipping plan with the northwest corner rule, then improve it                 |
| [`quadratic_programming/`](../quadratic_programming/core.py)               | Quadratic programming                     | Optimize an objective that includes quadratic terms under linear constraints                       |

Quadratic programming extends the linear objective to an expression such as:

$$
\min_x \left(c^T x + \tfrac{1}{2}x^T D x\right).
$$

The matrix $D$ describes the curvature of the objective. For a symmetric positive semidefinite $D$, this objective is convex: over a convex feasible set, a local minimum is also a global minimum. This mathematical property does not by itself establish correctness of the implementation.

## Getting started

Follow the [installation and execution instructions](../README.md#quick-start) in the README. All commands run from the repository root, even though this guide lives in `docs/`.

The console trace shows intermediate matrices, vectors, and algorithm steps. Useful Russian labels include `???` (step), `????????` (iteration), and `?????` (answer).

To try different data, edit the example block near the bottom of the corresponding file, under `if __name__ == '__main__':` or in the `simplex()` function it calls. Start with the existing data before changing dimensions or basis indices.

## Worked example: planning deliveries

The built-in [transportation example](../matrix_transport_problem/main.py) allocates shipments from three suppliers to three destinations. Its inputs are:

| Supplier            | Cost to destination 1 | Cost to destination 2 | Cost to destination 3 | Available supply |
| ------------------- | --------------------: | --------------------: | --------------------: | ---------------: |
| 1                   |                     8 |                     4 |                     1 |              100 |
| 2                   |                     8 |                     4 |                     3 |              300 |
| 3                   |                     9 |                     7 |                     5 |              300 |
| **Required demand** |               **300** |               **200** |               **200** |          **700** |

Costs are per shipped unit in arbitrary monetary units. Total supply and demand are both 700, so the problem is **balanced**.

Let $x_{ij}$ be the number of units shipped from supplier $i$ to destination $j$. We want to minimize:

$$
\sum_{i=1}^{3}\sum_{j=1}^{3} c_{ij}x_{ij}.
$$

Every supplier must ship its available supply, every destination must receive its required demand, and every shipment must be nonnegative. In matrix terms, each row sum must equal that supplier's supply, and each column sum must equal that destination's demand.

The implementation has two stages:

1. **Build a feasible plan.** The northwest corner rule starts in the upper-left cell, ships as much as possible, and moves on when a supply or demand is exhausted. It does not consider shipping costs.
2. **Improve the plan.** The next stage uses costs to search for a cheaper allocation while maintaining the required totals.

For these inputs, the northwest corner rule gives:

```text
                Destination
                 1    2    3
Supplier 1     100    0    0
Supplier 2     200  100    0
Supplier 3       0  100  200
```

Its total cost is $100(8) + 200(8) + 100(4) + 100(7) + 200(5) = 4500$.

For comparison, the following is another feasible plan:

```text
                Destination
                 1    2    3
Supplier 1       0    0  100
Supplier 2       0  200  100
Supplier 3     300    0    0
```

Its cost is $100(1) + 200(4) + 100(3) + 300(9) = 3900$. Checking its row and column sums confirms feasibility. This hand-calculated comparison illustrates why finding a feasible plan is only the first step; it is not a recorded execution result or a correctness test of the solver.

When reading the script's result, check all three things: supply totals, demand totals, and total cost. A smaller cost is meaningful only if the plan still satisfies the constraints.

For unbalanced inputs, the script adds a dummy supplier or destination with zero shipping costs. This balances the mathematics; in a real application, the meaning and cost of unmet demand or unused supply would need to be chosen explicitly.

## Understanding simplex and the code notation

For a problem with $m$ independent equality constraints, a **basis** selects $m$ columns of $A$ that form an invertible matrix, $A_B$. Setting the other variables to zero lets us solve for the basic variables. If these values are nonnegative, we have a **basic feasible solution**.

Geometrically, such solutions correspond to corners of the feasible region. Simplex changes one basis column at a time, seeking an improved objective value while preserving feasibility. A change of basis is called a **pivot**; in a degenerate case, it may leave the solution and objective unchanged.

The main phase assumes a suitable starting solution is already available. The initial phase introduces artificial variables to construct an auxiliary problem whose purpose is to find that starting point. It is a feasibility stage, not the optimization of the original objective.

Dual simplex approaches the problem from the other direction: its starting basis satisfies the dual optimality conditions, but some basic variables may violate nonnegativity. Its pivots aim to restore feasibility while retaining those dual conditions.

| Code notation          | Meaning                                                      |
| ---------------------- | ------------------------------------------------------------ |
| `m`, `n`               | Number of constraints and variables                          |
| `matrix_a`, `vector_b` | Constraint coefficients and right-hand-side values           |
| `vector_c`, `vector_x` | Objective coefficients and current solution                  |
| `vector_jb`, `B`       | Indices of the current basic variables                       |
| `matrix_ab`            | Columns of `A` selected by the basis                         |
| `vector_u`             | Simplex multipliers, also called potentials                  |
| `vector_delta`         | Reduced-cost quantities used to assess possible improvements |
| `vector_theta`         | Candidate step sizes used to choose a leaving basic variable |

Notation is local to each method: the transportation example uses `a` and `b` for supply and demand, `c` for a cost matrix, and basis entries for selected cells, which may include zero shipments.

**Indexing requires care.** The simplex and dual simplex examples enter basis indices starting at 1 and convert them to Python's zero-based indices. The quadratic programming code keeps one-based basis labels and subtracts 1 when accessing arrays. Check the relevant method before reusing its inputs.

## Suggested learning order

1. **Run the transportation example.** Relate each input and output to a shipment, then check the totals and cost by hand.
2. **Review the main simplex phase.** Focus on what a basis is, why it changes, and how a step size preserves feasibility.
3. **Study the inverse update.** Connect replacing a basis column to updating its inverse.
4. **Read the initial simplex phase.** Learn how a starting feasible basis can be constructed when one is not supplied.
5. **Explore dual simplex.** Compare its starting assumptions with those of the main simplex phase.
6. **Move to quadratic programming.** Examine how curvature changes the objective and the search direction.

For each method, try to answer: What does it assume at the start? What changes during an iteration? What stays valid? What makes it stop?

## Current limitations and next steps

- The repository contains standalone scripts and duplicated helpers rather than an installable package.
- Example data is embedded in source files; there is no shared command-line interface or input-file format.
- NumPy is pinned in `requirements.txt`. The six examples have been run with Python 3.14.3 and NumPy 2.5.3 on Windows; other environments remain untested.
- Eight automated regression tests cover the original examples and two applied cases. Broader inputs and reference-solver comparisons remain future work.
- Numerical tolerances, degenerate cases, termination behavior, and failure reporting need review before relying on results for unfamiliar inputs.
- The initial-phase entry point demonstrates the feasibility stage; it does not use its local `c` variable to optimize the original objective afterward.

Possible future work includes tests for more inputs and failure cases, plus comparisons with a reference solver.

## Reading the reported transportation result

The [formal case study](transportation-case-study.md) presents the example for reviewers. This section explains how to read the output yourself.

Your reported matrix has four positive entries. Read `X[0, 2] = 100` as ?warehouse 1 ships 100 units to destination 3.? Python indices start at zero. The four actual shipments are:

- Warehouse 1 to destination 3: 100 units.
- Warehouse 2 to destination 2: 200 units.
- Warehouse 2 to destination 3: 100 units.
- Warehouse 3 to destination 1: 300 units.

Multiplying these quantities by the corresponding route costs gives $100(1)+200(4)+100(3)+300(9)=3900$. Compare this with 4500 for the initial plan: the saving is 600, or about 13.3% of the initial cost.

### Why does B contain five cells but only four shipments?

For a balanced transportation problem with $m$ warehouses and $n$ destinations, a basis contains $m+n-1$ cells. One supply/demand equality is redundant because the grand totals agree, leaving five independent equalities in this 3-by-3 example.

The final `B = [(1, 1), (2, 2), (0, 2), (2, 0), (1, 2)]` uses zero-based indices. The cell `(2, 2)` means warehouse 3 to destination 3, and its shipment is zero. It remains in the basis to maintain the required basis structure. A basic solution with a zero basic variable is called **degenerate**; this alone is not an error.

The script prints the initial basis with indices shifted to start at 1, but prints the final basis with zero-based indices. This display difference does not change the shipment matrix.

### How do we know 3900 is the best possible cost?

Checking row and column totals proves feasibility. Finding a cheaper plan proves improvement. To prove optimality, we also need to show that no feasible plan can cost less.

The case study assigns a number to each warehouse and destination, called a potential. The sum of the two potentials for a route must not exceed that route's actual cost. This makes the resulting total a lower bound on the cost of every feasible plan.

With warehouse potentials $(0,2,4)$ and destination potentials $(5,2,1)$, that bound is:

$$
100(0)+300(2)+300(4)+300(5)+200(2)+200(1)=3900.
$$

Our feasible allocation costs exactly 3900, and the bound says nothing feasible can cost less. Together, those facts prove optimality for this example. The potentials are a certificate we can check independently of the algorithm's printed claim.

## Reproducing the original examples

All six entry points completed successfully on Windows using Python 3.14.3 and NumPy 2.5.3. Install the recorded dependency with `python -m pip install -r requirements.txt` inside the virtual environment. Pinning means selecting the exact NumPy version used for this check, so future installations do not silently choose a different release.

| Example               | Observed result                             | How to interpret it                                                                      |
| --------------------- | ------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Matrix inverse update | Rows `(1, 1, -1)`, `(0, 1, 0)`, `(0, 0, 1)` | The inverse after replacing the selected column                                          |
| Main simplex phase    | `x = (3, 2, 2, 0, 0)`                       | The reported optimal decision vector; with the supplied coefficients, the objective is 5 |
| Initial simplex phase | `x = (0, 0, 0)`                             | A feasible starting solution; this stage does not optimize the original objective        |
| Dual simplex          | `x = (0.25, 0.5, 0, 0, 0)`                  | The reported optimal decision vector; the supplied objective coefficients give -2.5      |
| Transportation        | Shipments documented in the case study      | Total cost 3900, with feasibility and optimality checked separately                      |
| Quadratic programming | `x = (1.7, 2.4, 0, 0.3)`                    | The reported optimal vector for the embedded quadratic example                           |

These results were initially recorded from execution. The regression tests described below now also check mathematical properties of these examples; they do not cover every possible input.

### Why did the quadratic example need a change?

The original code called `np.row_stack`, which is unavailable in the installed NumPy version. Replacing its two uses with `np.vstack` stacks the same two-dimensional blocks vertically. This is a compatibility repair; the mathematical steps and example inputs are unchanged.

This illustrates the distinction between an algorithm and its software environment: a previously working program can stop running because a dependency changes. Recording the environment makes such problems easier to reproduce.

For captured or redirected Russian-language output on Windows, use `python -X utf8 path/to/main.py` if the terminal's default encoding cannot represent the characters.

## What the regression tests establish

Run `python -m unittest discover -s tests -v` from the repository root with the environment activated. A regression test repeats a known case and checks that later changes preserve the required behavior. The tests live in [tests/test_examples.py](../tests/test_examples.py).

A program finishing without an error does not establish that its answer is valid. Our checks ask mathematical questions about returned values:

| Method                | What the test checks                                                                                 |
| --------------------- | ---------------------------------------------------------------------------------------------------- |
| Inverse update        | Multiplying the updated matrix by its computed inverse gives the identity in both orders             |
| Main and dual simplex | The solution satisfies the original equalities and nonnegativity; a dual bound matches its objective |
| Initial simplex phase | The function returns a feasible solution and an independent basis containing only original variables |
| Transportation        | Supply and demand totals, nonnegative shipments, cost 3900, and a matching lower bound               |
| Quadratic programming | Feasibility, convexity, and optimality conditions at the returned point; objective -19.95            |

The tests allow tiny numerical differences because floating-point calculations are approximate. They do not require a particular basis order or exact console wording: those can change without changing the mathematics.

### The return-value bug these tests found

The initial-phase example contains two constraints, where the second is twice the first. It therefore needs to remove a redundant constraint when constructing a basis. A nested function calls itself to finish this cleanup.

Previously, the inner call printed the correct answer, but the outer call did not return it. In Python, a function that reaches its end without a `return` statement returns `None`. That explains how the command-line example could look successful while another function trying to use its result would fail.

The repair adds `return` before the recursive call in both copies of the initial-phase implementation. The regression test verifies the actual returned solution and basis, so losing that return value again will fail the test.

### Why the quadratic test checks more than a saved answer

For the quadratic objective, the gradient is $c+Dx$. Equality-constraint multipliers account for directions that would violate the constraints. After accounting for these multipliers, the reduced gradient must be nonnegative, and each positive decision variable must have zero reduced gradient. The latter condition is called **complementarity**.

The test checks these conditions and that the quadratic matrix is positive semidefinite. Together with feasibility, they establish global optimality for this convex example. The computed objective is $-29.8+9.85=-19.95$.

Passing this suite gives us protection while cleaning up the repository. It does not yet establish behavior on infeasible, unbounded, unusually scaled, or arbitrary new problems.

## Reusing earlier assignments as modules

Each algorithm directory represents a separate university task. At the same time, later optimization methods naturally build on earlier ones. The repository now expresses both ideas: each task keeps its own folder and example, while reusable algorithms are imported from the task where they were introduced.

The matrix-inversion assignment owns [matrix_inversion/core.py](../matrix_inversion/core.py). It contains the inverse update and its supporting functions. [matrix_inversion/main.py](../matrix_inversion/main.py) remains the runnable example for that assignment.

The primal-simplex assignment follows the same pattern. [simplex_method/core.py](../simplex_method/core.py) contains the algorithm, while [simplex_method/main.py](../simplex_method/main.py) runs its original example. The initial-phase and dual-simplex tasks import these earlier modules instead of carrying private copies.

```text
matrix_inversion/core.py
          |
          v
simplex_method/core.py
          |
          +--------------------+
          v                    v
initial_stage_simplex_method   dual_simplex_method
```

Files named `__init__.py` make `matrix_inversion` and `simplex_method` Python packages. A package is a directory that can expose reusable modules. The `core.py` files contain algorithms; the `main.py` files contain the assignment demonstrations. This distinction makes it clear where to read the method and where to run it.

The folder hierarchy therefore still tells the academic story: matrix inversion was one task, primal simplex another, followed by the initial and dual methods. Reuse shows how those tasks depend on one another rather than presenting them as unrelated programs.

The examples can be run as modules from the repository root, for example `python -m initial_stage_simplex_method.main`.

Regression tests exercise each reusable `core.py` module. All six original examples also run through their package entry points.
