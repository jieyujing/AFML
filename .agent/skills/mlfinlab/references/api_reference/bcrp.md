# API Reference: bcrp.py

**Language**: Python

**Source**: `online_portfolio_selection/bcrp.py`

---

## Classes

### BCRP

This class implements the Best Constant Rebalanced Portfolio strategy. It is reproduced with
modification from the following paper:
`Li, B., Hoi, S. C.H., 2012. OnLine Portfolio Selection: A Survey. ACM Comput.
Surv. V, N, Article A (December 2012), 33 pages. <https://arxiv.org/abs/1212.2129>`_

Best Constant Rebalanced Portfolio rebalances to a set of weight that maximizes returns over a
given time period. This strategy is implemented in hindsight and is not predictive.

**Inherits from**: OLPS

#### Methods

##### _first_weight(self, weights)

Returns the first weight of the given portfolio to be the Best Constant Rebalanced Portfolio
in hindsight.

:param weights: (np.array) Given weights by the user.
:return: (np.array) Weights that maximize the returns.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| weights | None | - | - |



