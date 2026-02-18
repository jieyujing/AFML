# API Reference: test_bet_sizing.py

**Language**: Python

**Source**: `tests/test_bet_sizing.py`

---

## Classes

### TestBetSizeProbability

Tests the 'bet_size_probability' function.

**Inherits from**: unittest.TestCase

#### Methods

##### test_bet_size_probability_default(self)

Tests for successful execution using the default arguments of 'bet_size_probability', which are:
 average_active = False
 step_size = 0.0

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_bet_size_probability_avg_active(self)

Tests for successful execution of 'bet_size_probability' with 'average_active' set to True.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_bet_size_probability_stepsize(self)

Tests for successful execution of 'bet_size_probability' with 'step_size' greater than 0.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |




### TestBetSizeDynamic

Tests the 'bet_size_dynamic' function.

**Inherits from**: unittest.TestCase

#### Methods

##### test_bet_size_dynamic_default(self)

Tests for successful execution using the default arguments of 'bet_size_dynamic', which are:
 average_active = False
 step_size = 0.0

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |




### TestBetSizeBudget

Tests the 'bet_size_budget' function.

**Inherits from**: unittest.TestCase

#### Methods

##### test_bet_size_budget_default(self)

Tests for the successful execution of the 'bet_size_budget' function.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_bet_size_budget_div_zero(self)

Tests for successful handling of events DataFrames that result in a maximum number of
concurrent bet sides of zero.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |




### TestBetSizeReserve

Tests the 'bet_size_reserve' function.

**Inherits from**: unittest.TestCase

#### Methods

##### test_bet_size_reserve_default(self, mock_likely_parameters)

Tests for successful execution of 'bet_size_reserve' using default arguments, return_parameters=False.
Function 'most_likely_parameters' needs to be patched because the 'M2N.mp_fit' method makes use of
random numbers.

**Decorators**: `@patch('mlfinlab.bet_sizing.bet_sizing.most_likely_parameters')`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| mock_likely_parameters | None | - | - |


##### test_bet_size_reserve_return_params(self, mock_likely_parameters)

Tests for successful execution of 'bet_size_reserve' using return_parameters=True.
Function 'most_likely_parameters' needs to be patched because the 'M2N.mp_fit' method makes use of
random numbers.

**Decorators**: `@patch('mlfinlab.bet_sizing.bet_sizing.most_likely_parameters')`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| mock_likely_parameters | None | - | - |




### TestConfirmAndCastToDf

Tests the 'confirm_and_cast_to_df' function.

**Inherits from**: unittest.TestCase

#### Methods

##### test_cast_to_df_all_series(self)

Tests for successful execution of 'confirm_and_cast_to_df' when all dictionary values are pandas.Series.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_cast_to_df_one_series(self)

Tests for successful execution of 'confirm_and_cast_to_df' when only one of the dictionary values are
a pandas.Series.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_cast_to_df_no_series(self)

Tests for successful execution of 'confirm_and_cast_to_df' when none of the dictionary values are
a pandas.Series.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |




### TestGetConcurrentSides

Tests the function 'get_concurrent_sides' for successful operation.

**Inherits from**: unittest.TestCase

#### Methods

##### test_get_concurrent_sides_default(self)

Tests for the successful execution of 'get_concurrent_sides'. Since there are no options or branches,
there are no additional test cases beyond default.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |




### TestCdfMixture

Tests the 'cdf_mixture' function.

**Inherits from**: unittest.TestCase

#### Methods

##### test_cdf_mixture_default(self)

Tests for the successful execution of the 'cdf_mixture' function. Since there are no options or branches,
there are no additional test cases beyond default.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |




### TestSingleBetSizeMixed

Tests the 'single_bet_size_mixed' function.

**Inherits from**: unittest.TestCase

#### Methods

##### test_single_bet_size_mixed_above_zero(self)

Tests for the successful execution of the 'single_bet_size_mixed' function where the 'c_t' parameter is
greater than zero.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_single_bet_size_mixed_below_zero(self)

Tests for the successful execution of the 'single_bet_size_mixed' function where the 'c_t' parameter is
less than zero.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |



