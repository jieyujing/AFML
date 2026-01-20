# API Reference: test_cla.py

**Language**: Python

**Source**: `tests/test_cla.py`

---

## Classes

### TestCLA

Tests different functions of the CLA class.

**Inherits from**: unittest.TestCase

#### Methods

##### setUp(self)

Set the file path for the tick data csv.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_cla_with_mean_returns(self)

Test the calculation of CLA turning points using mean returns.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_cla_with_weight_bounds_as_lists(self)

Test the calculation of CLA turning points when we pass the weight bounds as a list
instead of just lower and upper bound value.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_cla_with_exponential_returns(self)

Test the calculation of CLA turning points using exponential returns

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_cla_max_sharpe(self)

Test the calculation of maximum sharpe ratio weights.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_cla_min_volatility(self)

Test the calculation for minimum volatility weights.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_cla_efficient_frontier(self)

Test the calculation of the efficient frontier solution.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_lambda_for_no_bounded_weights(self)

Test the computation of lambda when there are no bounded weights.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_free_bound_weights(self)

Test the method of freeing bounded weights when free-weights is None.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_expected_returns_equals_means(self)

Test for condition when expected returns equal the mean value.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_lambda_for_zero_matrices(self)

Test the computation of lambda when there are no bounded weights. The method
should return None, None.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_w_for_no_bounded_weights(self)

Test the computation of weights (w) when there are no bounded weights.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_purge_excess(self)

Test purge number excess for very very small tolerance.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_flag_true_for_purge_num_err(self)

Test whether the flag becomes True in the purge num error function.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_value_error_for_unknown_solution(self)

Test ValueError on passing unknown solution string.

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


##### test_value_error_for_unknown_returns(self)

Test ValueError on passing unknown returns string.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_resampling_asset_prices(self)

Test resampling of asset prices.

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


##### test_cla_with_input_as_returns_and_covariance(self)

Test CLA when we pass expected returns and covariance matrix as input.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_no_asset_names(self)

Test CLA when not supplying a list of asset names.

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



