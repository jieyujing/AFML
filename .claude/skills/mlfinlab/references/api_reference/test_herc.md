# API Reference: test_herc.py

**Language**: Python

**Source**: `tests/test_herc.py`

---

## Classes

### TestHERC

Tests different functions of the HERC algorithm class.

**Inherits from**: unittest.TestCase

#### Methods

##### setUp(self)

Set the file path for the tick data csv

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_herc_equal_weight(self)

Test the weights calculated by the HERC algorithm - if all the weights are positive and
their sum is equal to 1.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_herc_min_variance(self)

Test the weights calculated by the HERC algorithm - if all the weights are positive and
their sum is equal to 1.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_herc_min_standard_deviation(self)

Test the weights calculated by the HERC algorithm - if all the weights are positive and
their sum is equal to 1.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_value_error_for_expected_shortfall(self)

Test ValueError when expected_shortfall is the allocation metric, no asset_returns dataframe
is given and no asset_prices dataframe is passed.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_herc_expected_shortfall(self)

Test the weights calculated by the HERC algorithm - if all the weights are positive and
their sum is equal to 1.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_herc_conditional_drawdown_risk(self)

Test the weights calculated by the HERC algorithm - if all the weights are positive and
their sum is equal to 1.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_quasi_diagnalization(self)

Test the quasi-diagnalisation step of HERC algorithm.

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


##### test_herc_with_input_as_returns(self)

Test HERC when passing asset returns dataframe as input.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_herc_with_asset_returns_as_none(self)

Test HERC when asset returns are not required for calculating the weights.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_herc_with_input_as_covariance_matrix(self)

Test HERC when passing a covariance matrix as input.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_value_error_for_risk_measure(self)

Test HERC when a different allocation metric string is used.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_no_asset_names(self)

Test HERC when not supplying a list of asset names.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_no_asset_names_with_asset_returns(self)

Test HERC when not supplying a list of asset names and when the user passes asset_returns.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_value_error_with_no_asset_names(self)

Test ValueError when not supplying a list of asset names and no other input

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



