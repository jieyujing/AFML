# API Reference: test_exponential_gradient.py

**Language**: Python

**Source**: `tests/test_exponential_gradient.py`

---

## Classes

### TestExponentialGradient

Tests different functions of the Exponential Gradient class.

**Inherits from**: TestCase

#### Methods

##### setUp(self)

Sets the file path for the tick data csv.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_mu_solution(self)

Test calculation of exponential gradient weights with multiplicative update rule.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_gp_solution(self)

Test calculation of exponential gradient weights with gradient projection update rule.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_em_solution(self)

Test calculation of exponential gradient weights with expectation maximization update rule.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_wrong_update(self)

Tests ValueError if the passing update rule is not correct.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |



