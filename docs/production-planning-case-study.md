# Case Study: Production Planning

This illustrative example chooses weekly production quantities for two products. Product A earns 40 units of profit and product B earns 30.

The available resources give the following model:

$$
\max 40x_A+30x_B
$$

subject to:

$$
2x_A+x_B\leq100,
$$

$$
x_A+1.5x_B\leq90,
$$

$$
x_A\leq40, \qquad x_A,x_B\geq0.
$$

The first two constraints represent two limited resources. The third is a separate capacity limit for product A. Slack variables convert the inequalities to the equality form used by the simplex implementation.

## Result

Run the example from the repository root:

```sh
python -m cases.production_planning
```

The solution is:

| Decision | Value |
| --- | ---: |
| Product A | 30 |
| Product B | 40 |
| Maximum profit | 2400 |

Both shared resources are fully used:

$$
2(30)+40=100, \qquad 30+1.5(40)=90.
$$

The separate capacity for product A has 10 unused units.

## Optimality check

Dual resource values of 15 and 10, with zero value for the unused product-A capacity, satisfy:

$$
2(15)+10=40, \qquad 15+1.5(10)=30.
$$

Their bound is $100(15)+90(10)=2400$, equal to the plan's profit. This proves optimality for the example.

The figures are synthetic and are included to demonstrate formulation and interpretation rather than describe a real factory.
