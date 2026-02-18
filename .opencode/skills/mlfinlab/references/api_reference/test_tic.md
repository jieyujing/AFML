# API Reference: test_tic.py

**Language**: Python

**Source**: `tests/test_tic.py`

---

## Classes

### TestTIC

Tests different functions of the TIC algorithm class.

**Inherits from**: unittest.TestCase

#### Methods

##### setUp(self)

Initialize and load data

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_get_linkage_corr(self)

Testing the creation of a linkage object from empirical correlation matrix and tree graph

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_link_clusters()

Testing the transformation of linkage object from local linkage to global linkage

**Decorators**: `@staticmethod`


##### test_update_dist()

Testing the update of the general distance matrix to take the new clusters into account

**Decorators**: `@staticmethod`


##### test_get_atoms()

Testing the obtaining of the atoms included in an element from a linkage object

**Decorators**: `@staticmethod`


##### test_link2corr(self)

Test the process of deriving a correlation matrix from the linkage object

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_tic_correlation(self)

Test the calculation the Theory-Implies Correlation (TIC) matrix.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_corr_dist(self)

Test the calculation of the correlation matrix distance

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |



