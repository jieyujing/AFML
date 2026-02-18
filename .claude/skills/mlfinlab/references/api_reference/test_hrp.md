# API Reference: test_hrp.py

**Language**: Python

**Source**: `tests/test_hrp.py`

---

## Classes

### TestHRP

Tests different functions of the HRP algorithm class.

**Inherits from**: unittest.TestCase

#### Methods

##### setUp(self)

Set the file path for the tick data csv.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_hrp(self)

Test the weights calculated by the HRP algorithm - if all the weights are positive and
their sum is equal to 1.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_hrp_long_short(self)

Test the Long Short Portfolio via side_weights Serries 1 for Long, -1 for Short (index=asset names)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_dendrogram_plot(self)

Test if dendrogram plot object is correctly rendered.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_quasi_diagnalization(self)

Test the quasi-diagnalisation step of HRP algorithm.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_value_error_for_non_dataframe_input(self)

Test ValueError on passing non-dataframe input.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_value_error_for_non_date_index(self)

Test ValueError on passing dataframe not indexed by date.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_all_inputs_none(self)

Test allocation when all inputs are None.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_hrp_with_input_as_returns(self)

Test HRP when passing asset returns dataframe as input.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_hrp_with_input_as_covariance_matrix(self)

Test HRP when passing a covariance matrix as input.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_hrp_with_input_as_distance_matrix(self)

Test HRP when passing a distance matrix as input.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_hrp_with_linkage_method(self)

Test HRP when passing a custom linkage method.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_no_asset_names(self)

Test HRP when not supplying a list of asset names.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_no_asset_names_with_asset_returns(self)

Test HRP when not supplying a list of asset names and when the user passes asset_returns.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_valuerror_with_no_asset_names(self)

Test ValueError when not supplying a list of asset names and no other input.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |



