# API Reference: test_nco.py

**Language**: Python

**Source**: `tests/test_nco.py`

---

## Classes

### TestNCO

Tests different functions of the NCO algorithm class.

**Inherits from**: unittest.TestCase

#### Methods

##### setUp(self)

Initialize

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_simulate_covariance()

Test the deriving an empirical vector of means and an empirical covariance matrix.

**Decorators**: `@staticmethod`


##### test_cluster_kmeans_base(self)

Test the finding of the optimal partition of clusters using K-Means algorithm.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_allocate_cvo()

Test the estimates of the Convex Optimization Solution (CVO).

**Decorators**: `@staticmethod`


##### test_allocate_nco()

Test the estimates the optimal allocation using the (NCO) algorithm

**Decorators**: `@staticmethod`


##### test_allocate_mcos()

Test the estimates of the optimal allocation using the Monte Carlo optimization selection

**Decorators**: `@staticmethod`


##### test_estim_errors_mcos()

Test the computation the true optimal allocation w, and compares that result with the estimated ones by MCOS.

**Decorators**: `@staticmethod`


##### test_form_block_matrix()

Test the creation of a block correlation matrix with given parameters.

**Decorators**: `@staticmethod`


##### test_form_true_matrix()

Test the creation of a random vector of means and a random covariance matrix.

**Decorators**: `@staticmethod`



