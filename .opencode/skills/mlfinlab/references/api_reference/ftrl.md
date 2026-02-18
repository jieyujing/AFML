# API Reference: ftrl.py

**Language**: Python

**Source**: `online_portfolio_selection/ftrl.py`

---

## Classes

### FTRL

This class implements the Follow the Regularized Leader strategy. It is reproduced with
modification from the following paper:
`Li, B., Hoi, S. C.H., 2012. OnLine Portfolio Selection: A Survey. ACM Comput.
Surv. V, N, Article A (December 2012), 33 pages. <https://arxiv.org/abs/1212.2129>`_

Follow the Regularized Leader strategy directly tracks the Best Constant Rebalanced Portfolio
until the previous period with an additional regularization term

**Inherits from**: FTL

#### Methods

##### __init__(self, beta)

Initializes Follow the Regularized Leader with a beta constant term.

:param beta: (float) Constant to the regularization term. Typical ranges for interesting
                     results include [0, 0.2], 1, and any high values. Low beta
                     FTRL strategies are identical to FTL, and high beta indicates more
                     regularization to return a uniform CRP.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| beta | None | - | - |


##### _fast_optimize(self, optimize_array)

Calculates weights that maximize returns over the given array.

:param optimize_array: (np.array) Relative returns of the assets for a given time period.
:return: (np.array) Weights that maximize the returns for the given array.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| optimize_array | None | - | - |




## Functions

### _objective(weight)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| weight | None | - | - |

**Returns**: (none)


