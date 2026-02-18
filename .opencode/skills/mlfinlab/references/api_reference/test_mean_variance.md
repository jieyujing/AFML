# API Reference: test_mean_variance.py

**Language**: Python

**Source**: `tests/test_mean_variance.py`

---

## Classes

### TestMVO

Tests the different functions of the Mean Variance Optimisation class

**Inherits from**: unittest.TestCase

#### Methods

##### setUp(self)

Set the file path for the tick data csv.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_inverse_variance_solution(self)

Test the calculation of inverse-variance portfolio weights.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_min_volatility_solution(self)

Test the calculation of minimum volatility portfolio weights.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_max_return_min_volatility_solution(self)

Test the calculation of maximum expected return and minimum volatility portfolio weights.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_max_sharpe_solution(self)

Test the calculation of maximum sharpe portfolio weights.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_min_volatility_for_target_return(self)

Test the calculation of minimum volatility-target return portfolio weights.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_max_return_for_target_risk(self)

Test the calculation of maximum return-target risk portfolio weights.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_max_diversification(self)

Test the calculation of maximum diversification portfolio weights.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_max_decorrelation(self)

Test the calculation of maximum decorrelation portfolio weights.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_plotting_efficient_frontier(self)

Test the plotting of the efficient frontier.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_exception_in_plotting_efficient_frontier(self)

Test raising of exception when plotting the efficient frontier.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_mvo_with_input_as_returns_and_covariance(self)

Test MVO when we pass expected returns and covariance matrix as input.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_min_volatility_with_specific_weight_bounds(self)

Test the calculation of weights when specific bounds are supplied.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_max_sharpe_with_specific_weight_bounds(self)

Test the calculation of weights when specific bounds are supplied.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_efficient_risk_with_specific_weight_bounds(self)

Test the calculation of weights when specific bounds are supplied.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_efficient_return_with_specific_weight_bounds(self)

Test the calculation of weights when specific bounds are supplied.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_max_decorrelation_with_specific_weight_bounds(self)

Test the calculation of weights when specific bounds are supplied.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_max_diversification_with_specific_weight_bounds(self)

Test the calculation of weights when specific bounds are supplied.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_max_return_min_volatility_with_specific_weight_bounds(self)

Test the calculation of weights when specific bounds are supplied.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_mvo_with_exponential_returns(self)

Test the calculation of inverse-variance portfolio weights.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_unknown_returns_calculation(self)

Test ValueError on passing unknown returns calculation string.

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


##### test_value_error_for_no_min_volatility_optimal_weights(self)

Test ValueError when no optimal weights are found for minimum volatility solution.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_value_error_for_no_quadratic_utlity_optimal_weights(self)

Test ValueError when no optimal weights are found for max return-minimum volatility solution.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_value_error_for_no_max_sharpe_optimal_weights(self)

Test ValueError when no optimal weights are found for maximum Sharpe solution.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_value_error_for_no_efficient_risk_optimal_weights(self)

Test ValueError when no optimal weights are found for efficient risk solution.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_value_error_for_no_efficient_return_optimal_weights(self)

Test ValueError when no optimal weights are found for efficient return solution.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_value_error_for_no_max_diversification_optimal_weights(self)

Test ValueError when no optimal weights are found for max diversification solution.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_value_error_for_no_max_decorrelation_optimal_weights(self)

Test ValueError when no optimal weights are found for max decorrelation solution.

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


##### test_no_asset_names(self)

Test MVO when not supplying a list of asset names.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_no_asset_names_by_passing_cov(self)

Test MVO when not supplying a list of asset names but passing covariance matrix as input

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_valuerror_with_no_asset_names(self)

Test ValueError when not supplying a list of asset names and no other input

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_portfolio_metrics(self)

Test the printing of portfolio metrics to stdout.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_custom_objective_function(self)

Test custom portfolio objective and allocation constraints.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_custom_objective_with_asset_names(self)

Test custom portfolio objective and constraints while providing a list of asset names.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_value_error_for_custom_obj_optimal_weights(self)

Test ValueError when no optimal weights are found for custom objective solution.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |



