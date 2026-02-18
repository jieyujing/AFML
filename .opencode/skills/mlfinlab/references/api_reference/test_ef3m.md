# API Reference: test_ef3m.py

**Language**: Python

**Source**: `tests/test_ef3m.py`

---

## Classes

### TestM2NConstructor

Tests the constructor method of the M2N class.

**Inherits from**: unittest.TestCase

#### Methods

##### test_m2n_constructor(self)

Tests that the constructor of the M2N class executes properly.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |




### TestM2NGetMoments

Tests the 'get_moments' method of the M2N class.

**Inherits from**: unittest.TestCase

#### Methods

##### test_get_moments(self)

Tests the 'get_moments' method of the M2N class.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |




### TestM2NIter4

Tests the 'iter_4' method of the M2N class.

**Inherits from**: unittest.TestCase

#### Methods

##### test_iter_4_validity_check_1(self)

Tests 'iter_4' method's 'Validity check 1' breakpoint condition.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_iter_4_validity_check_2(self)

Tests 'iter_4' method's 'Validity check 2' breakpoint condition.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_iter_4_validity_check_3(self)

Tests 'iter_4' method's 'Validity check 3' breakpoint condition.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_iter_4_validity_check_5(self)

Tests 'iter_4' method's 'Validity check 5' breakpoint condition.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_iter_4_validity_check_6(self)

Tests 'iter_4' method's 'Validity check 6' breakpoint condition.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_iter_4_success(self)

Tests 'iter_4' method for successful execution.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |




### TestM2NIter5

Tests the 'iter_5' method of the M2N class.

**Inherits from**: unittest.TestCase

#### Methods

##### test_iter_5_validity_check_1(self)

Tests 'iter_5' method's 'Validity check 1' breakpoint condition.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_iter_5_validity_check_2(self)

Tests 'iter_5' method's 'Validity check 2' breakpoint condition.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_iter_5_validity_check_3(self)

Tests 'iter_5' method's 'Validity check 3' breakpoint condition.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_iter_5_validity_check_5(self)

Tests 'iter_5' method's 'Validity check 5' breakpoint condition.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_iter_5_validity_check_6(self)

Tests 'iter_5' method's 'Validity check 6' breakpoint condition.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_iter_5_validity_check_7(self)

Tests 'iter_5' method's 'Validity check 7' breakpoint condition.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_iter_5_validity_check_8(self)

Tests 'iter_5' method's 'Validity check 8' breakpoint condition.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_iter_5_validity_check_9(self)

Tests 'iter_5' method's 'Validity check 9' breakpoint condition.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_iter_5_success(self)

Tests 'iter_5' method for successful execution.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |




### TestM2NFit

Tets the 'fit' method of the M2N class.

**Inherits from**: unittest.TestCase

#### Methods

##### test_fit_variant_1(self)

Tests the 'fit' method of the M2N class, using variant 1.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_fit_variant_2(self)

Tests the 'fit' method of the M2N class, using variant 2.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_fit_variant_value_error(self)

Tests that the 'fit' method throws a ValueError if an invalid value is passed to argument 'variant'.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_fit_success_via_error(self)

Tests that the 'fit' method successfully exits due to a low error being reached.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_fit_success_via_epsilon(self)

Tests that the 'fit' method successfully exits due to p_1 converging.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_fit_success_via_max_iter(self)

Tests that the 'fit' method successfully exits due to the maximum number of iterations being reached.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |




### TestM2NEF3M

Tests the EF3M algorithms of the M2N module.

**Inherits from**: unittest.TestCase

#### Methods

##### test_ef3m_variant_1(self)

Tests the 'iter_4_jit' function of the M2N module (using variant 1).

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_ef3m_variant_2(self)

Tests the 'iter_5_jit' function of the M2N module (using variant 2).

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |




### TestM2NSingleFitLoop

Tests the 'single_fit_loop' method.

**Inherits from**: unittest.TestCase

#### Methods

##### test_single_fit_loop_return_type(self)

Tests that the 'single_fit_loop' method executes successfully.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |




### TestM2NMpFit

Tests the 'mp_fit' method.

**Inherits from**: unittest.TestCase

#### Methods

##### test_mp_fit_return_type(self)

Tests that the 'mp_fit' method executes successfully.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |




### TestCenteredMoment

Tests the helper function 'centered_moment'.

**Inherits from**: unittest.TestCase

#### Methods

##### test_centered_moment_result(self)

Tests for the successful execution of the 'centered_moment' helper function.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |




### TestRawMoment

Tests the helper function 'raw_moment'.

**Inherits from**: unittest.TestCase

#### Methods

##### test_raw_moment_result(self)

Tests for the successful execution of the 'raw_moment' helper function.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |




### TestMostLikelyParameters

Tests the helper function 'most_likely_parameters'.

**Inherits from**: unittest.TestCase

#### Methods

##### test_most_likely_parameters_result(self)

Tests for the successful execution of the 'most_likely_parameters' function.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_most_likely_parameters_list_arg(self)

Tests the helper function 'most_likely_parameters' when passing a list to 'ignore_columns'.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |



