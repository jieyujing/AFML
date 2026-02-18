# API Reference: test_ch10_snippets.py

**Language**: Python

**Source**: `tests/test_ch10_snippets.py`

---

## Classes

### TestCh10Snippets

Tests the following functions in ch10_snippets.py:
- get_signal
- avg_active_signals
- mp_avg_active_signals
- discrete_signal

**Inherits from**: unittest.TestCase

#### Methods

##### setUp(self)

Sets up the data to be used for the following tests.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_get_signal(self)

Tests calculating the bet size from probability.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_avg_active_signals(self)

Tests the avg_active_signals function. Also implicitly tests the
molecular multiprocessing function mp_avg_active_signals.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_mp_avg_active_signals(self)

An explicit test of the mp_avg_active_signals subroutine.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_discrete_signal(self)

Tests the discrete_signal function.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |




### TestBetSize

Test case for bet_size, bet_size_sigmoid, and bet_size_power.

**Inherits from**: unittest.TestCase

#### Methods

##### test_bet_size_sigmoid(self)

Tests successful execution of 'bet_size_sigmoid'.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_bet_size_power(self)

Tests successful execution of 'bet_size_power'.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_bet_size_power_value_error(self)

Tests successful raising of ValueError in 'bet_size_power'.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_bet_size_power_return_zero(self)

Tests that the function returns zero if the price divergence provided is zero.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_bet_size(self)

Test excution in all function modes of 'bet_size'.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_bet_size_key_error(self)

Tests for the KeyError in the event that an invalid function is provided to 'func'.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |




### TestGetTPos

Test case for get_target_pos, get_target_pos_sigmoid, and get_target_pos_power.

**Inherits from**: unittest.TestCase

#### Methods

##### test_get_target_pos_sigmoid(self)

Tests successful execution of 'get_target_pos_sigmoid'.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_get_target_pos_power(self)

Tests successful execution of 'get_target_pos_power'.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_get_target_pos(self)

Tests successful execution in 'sigmoid' and 'power' function variants of 'get_target_pos'.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_get_target_pos_key_error(self)

Tests for the KeyError in 'get_target_pos' in the case that an invalid value for 'func' is passed.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |




### TestInvPrice

Tests for functions 'inv_price', 'inv_price_sigmoid', and 'inv_price_power'.

**Inherits from**: unittest.TestCase

#### Methods

##### test_inv_price_sigmoid(self)

Test for the successful execution of 'inv_price_sigmoid'.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_inv_price_power(self)

Test for the successful execution of 'inv_price_sigmoid'.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_inv_price(self)

Test for successful execution of 'inv_price' function under different function options.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_inv_price_key_error(self)

Test for successful raising of KeyError in response to invalid choice of 'func' argument.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |




### TestLimitPrice

Tests the functions 'limit_price_sigmoid', 'limit_price_power', and 'limit_price'.

**Inherits from**: unittest.TestCase

#### Methods

##### test_limit_price_sigmoid(self)

Test successful execution of 'limit_price_sigmoid' function.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_limit_price_sigmoid_return_nan(self)

Tests for the successful return of np.nan in the case that the target position is the same as the
current position.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_limit_price_power(self)

Test successful execution of 'limit_price_power' function.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_limit_price_key_error(self)

Tests raising of the KeyError due to invalid choice of 'func' argument.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |




### TestGetW

Tests the functions 'get_w_sigmoid', 'get_w_power', and 'get_w'.

**Inherits from**: unittest.TestCase

#### Methods

##### test_get_w_sigmoid(self)

Tests successful execution of 'get_w_sigmoid' function.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_get_w_power(self)

Tests successful execution of 'get_w_power' function.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_get_w_power_value_error(self)

Tests that a ValueError is raised if the price divergence 'x' is not between -1 and 1, inclusive.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_get_w_power_warning(self)

Tests that a UserWarning is raised if 'w' is calcualted to be less than zero, and returns a zero.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_get_w_key_error(self)

Tests that a KeyError is raised if an invalid function is passed to argument 'func'.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |



