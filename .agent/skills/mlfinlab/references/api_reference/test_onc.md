# API Reference: test_onc.py

**Language**: Python

**Source**: `tests/test_onc.py`

---

## Classes

### TestOptimalNumberOfClusters

Test get_onc_clusters function

**Inherits from**: unittest.TestCase

#### Methods

##### setUp(self)

Set the file path for the sample dollar bars data.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### _check_if_in_cluster(array, cluster_dict)

Check if array is in dictionary values not taking into account order of elements

**Decorators**: `@staticmethod`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| array | None | - | - |
| cluster_dict | None | - | - |


##### test_get_onc_clusters(self)

Test get_onc_clusters function on Breast Cancer data set from sklearn

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_check_redo_condition(self)

Function to test redo condition helper function

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |



