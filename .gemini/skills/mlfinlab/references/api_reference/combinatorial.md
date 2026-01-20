# API Reference: combinatorial.py

**Language**: Python

**Source**: `cross_validation/combinatorial.py`

---

## Classes

### CombinatorialPurgedKFold

Advances in Financial Machine Learning, Chapter 12.

Implements Combinatial Purged Cross Validation (CPCV)

The train is purged of observations overlapping test-label intervals
Test set is assumed contiguous (shuffle=False), w/o training samples in between

:param n_splits: (int) The number of splits. Default to 3
:param samples_info_sets: (pd.Series) The information range on which each record is constructed from
    *samples_info_sets.index*: Time when the information extraction started.
    *samples_info_sets.value*: Time when the information extraction ended.
:param pct_embargo: (float) Percent that determines the embargo size.

**Inherits from**: KFold

#### Methods

##### __init__(self, n_splits: int = 3, n_test_splits: int = 2, samples_info_sets: pd.Series = None, embargo: int = 1)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| n_splits | int | 3 | - |
| n_test_splits | int | 2 | - |
| samples_info_sets | pd.Series | None | - |
| embargo | int | 1 | - |


##### _generate_combinatorial_test_ranges(self, splits_indices: dict) → List

Using start and end indices of test splits from KFolds and number of test_splits (self.n_test_splits),
generates combinatorial test ranges splits

:param splits_indices: (dict) Test fold integer index: [start test index, end test index]
:return: (list) Combinatorial test splits ([start index, end index])

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| splits_indices | dict | - | - |

**Returns**: `List`


##### _fill_backtest_paths(self, train_indices: list, test_splits: list)

Using start and end indices of test splits and purged/embargoed train indices from CPCV, find backtest path and
place in the path where these indices should be used.

:param test_splits: (list) of lists with first element corresponding to test start index and second - test end

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| train_indices | list | - | - |
| test_splits | list | - | - |


##### split(self, X: pd.DataFrame, y: pd.Series = None, groups = None)

The main method to call for the PurgedKFold class

:param X: (pd.DataFrame) Samples dataset that is to be split
:param y: (pd.Series) Sample labels series
:param groups: (array-like), with shape (n_samples,), optional
    Group labels for the samples used while splitting the dataset into
    train/test set.
:return: (tuple) [train list of sample indices, and test list of sample indices]

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| X | pd.DataFrame | - | - |
| y | pd.Series | None | - |
| groups | None | None | - |




## Functions

### _get_number_of_backtest_paths(n_train_splits: int, n_test_splits: int) → float

Number of combinatorial paths for CPCV(N,K)
:param n_train_splits: (int) number of train splits
:param n_test_splits: (int) number of test splits
:return: (int) number of backtest paths for CPCV(N,k)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| n_train_splits | int | - | - |
| n_test_splits | int | - | - |

**Returns**: `float`


