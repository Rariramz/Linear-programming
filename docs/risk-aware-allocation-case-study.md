# Case Study: Risk-Aware Allocation

This illustrative example distributes a budget of 100 units among three options. A linear term represents expected benefit, while a quadratic penalty discourages concentrating too much in options with larger risk coefficients.

The model is:

$$
\min_x \left(c^Tx+\tfrac12x^TDx\right)
$$

subject to $x_1+x_2+x_3=100$ and $x\geq0$, where:

$$
c=(-5,-6,-4), \qquad D=\operatorname{diag}(0.1,0.2,0.05).
$$

The second option has the strongest linear benefit and the largest quadratic penalty. The third has a weaker benefit but the smallest penalty.

## Result

Run the example from the repository root:

```sh
python -m cases.risk_aware_allocation
```

| Option | Allocation |
| --- | ---: |
| 1 | 32.86 |
| 2 | 21.43 |
| 3 | 45.71 |

The allocations sum to 100 and the objective value is approximately `-323.57`.

At the solution, each component of the gradient $c+Dx$ is approximately `-1.7143`. The equality-constraint multiplier offsets this common value, satisfying the stationarity condition. Since $D$ is positive definite and the plan is feasible, these conditions establish the global minimum.

This is a small synthetic allocation example. It omits correlations, changing returns, transaction costs, and other features required for a financial model.
