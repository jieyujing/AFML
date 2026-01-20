# API Reference: scorn.py

**Language**: Python

**Source**: `online_portfolio_selection/scorn.py`

---

## Classes

### SCORN

This class implements the Symmetric Correlation Driven Nonparametric Learning strategy. It is
reproduced with modification from the following paper:
`Wang, Y., & Wang, D. (2019). Market Symmetry and Its Application to Pattern-Matching-Based
Portfolio Selection. The Journal of Financial Data Science, 1(2), 78–92.
<https://jfds.pm-research.com/content/1/2/78>`_

Symmetric Correlation Driven Nonparametric Learning is an extension of the CORN strategy
proposed by Yang Wang and Dong Wang. SCORN looks to not only maximize the returns for the similar
periods but also minimize the losses from the negatively correlated periods.

**Inherits from**: CORN

#### Methods

##### _update_weight(self, time)

Predicts the next time's portfolio weight.

:param time: (int) Current time period.
:return: (np.array) Predicted weights.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| time | None | - | - |


##### _scorn_optimize(self, similar, opposite)

Calculates weights that maximize returns over the given array.

:param similar: (np.array) Relative returns of similar periods.
:param opposite: (np.array) Relative returns of inversely similar periods.
:return: (np.array) Weights that maximize the returns for the given array.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| similar | None | - | - |
| opposite | None | - | - |




## Functions

### _objective(weight)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| weight | None | - | - |

**Returns**: (none)



### _derivative(weight)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| weight | None | - | - |

**Returns**: (none)


