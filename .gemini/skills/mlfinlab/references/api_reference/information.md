# API Reference: information.py

**Language**: Python

**Source**: `codependence/information.py`

---

## Functions

### get_optimal_number_of_bins(num_obs: int, corr_coef: float = None) → int

Calculates optimal number of bins for discretization based on number of observations
and correlation coefficient (univariate case).

Algorithms used in this function were originally proposed in the works of Hacine-Gharbi et al. (2012)
and Hacine-Gharbi and Ravier (2018). They are described in the Cornell lecture notes:
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994&download=yes (p.26)

:param num_obs: (int) Number of observations.
:param corr_coef: (float) Correlation coefficient, used to estimate the number of bins for univariate case.
:return: (int) Optimal number of bins.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| num_obs | int | - | - |
| corr_coef | float | None | - |

**Returns**: `int`



### get_mutual_info(x: np.array, y: np.array, n_bins: int = None, normalize: bool = False) → float

Returns mutual information (I) between two vectors.

This function uses the discretization with the optimal bins algorithm proposed in the works of
Hacine-Gharbi et al. (2012) and Hacine-Gharbi and Ravier (2018).

Read Cornell lecture notes for more information about the mutual information:
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994&download=yes.

:param x: (np.array) X vector.
:param y: (np.array) Y vector.
:param n_bins: (int) Number of bins for discretization, if None the optimal number will be calculated.
                     (None by default)
:param normalize: (bool) Flag used to normalize the result to [0, 1]. (False by default)
:return: (float) Mutual information score.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| x | np.array | - | - |
| y | np.array | - | - |
| n_bins | int | None | - |
| normalize | bool | False | - |

**Returns**: `float`



### variation_of_information_score(x: np.array, y: np.array, n_bins: int = None, normalize: bool = False) → float

Returns variantion of information (VI) between two vectors.

This function uses the discretization using optimal bins algorithm proposed in the works of
Hacine-Gharbi et al. (2012) and Hacine-Gharbi and Ravier (2018).

Read Cornell lecture notes for more information about the variation of information:
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994&download=yes.

:param x: (np.array) X vector.
:param y: (np.array) Y vector.
:param n_bins: (int) Number of bins for discretization, if None the optimal number will be calculated.
                     (None by default)
:param normalize: (bool) True to normalize the result to [0, 1]. (False by default)
:return: (float) Variation of information score.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| x | np.array | - | - |
| y | np.array | - | - |
| n_bins | int | None | - |
| normalize | bool | False | - |

**Returns**: `float`


