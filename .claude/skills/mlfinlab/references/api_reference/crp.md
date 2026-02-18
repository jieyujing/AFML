# API Reference: crp.py

**Language**: Python

**Source**: `online_portfolio_selection/crp.py`

---

## Classes

### CRP

This class implements the Buy and Hold strategy. It is reproduced with modification from
the following paper:
`Li, B., Hoi, S. C.H., 2012. OnLine Portfolio Selection: A Survey. ACM Comput.
Surv. V, N, Article A (December 2012), 33 pages. <https://arxiv.org/abs/1212.2129>`_

Constant Rebalanced Portfolio rebalances to a given weight each time period.

**Inherits from**: OLPS

#### Methods

##### __init__(self, weight = None)

Sets the recurring weights for the Constant Rebalanced Portfolio. If weight is given,
this will override any given weights inputted by the user through ``allocate``.

:param weight: (list/np.array/pd.DataFrame) Initial weight set by the user.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| weight | None | None | - |


##### _first_weight(self, weights)

Sets first weight for Constant Rebalanced Portfolio

:param weights: (list/np.array/pd.DataFrame) initial weights set by the user.
:return: (np.array) First portfolio weight.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| weights | None | - | - |



