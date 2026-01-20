# API Reference: test_labeling_matrix_flags.py

**Language**: Python

**Source**: `tests/test_labeling_matrix_flags.py`

---

## Classes

### TestMatrixFlagLabels

Tests for the matrix flags labeling method.

**Inherits from**: unittest.TestCase

#### Methods

##### setUp(self)

Set the file path for data.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_init(self)

Tests that exceptions are raised correctly during initialization of the class if inputs are wrong.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_set_template(self)

Tests for user setting a new template. Also verifies that exception is raised for invalid template formats.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_transform_data(self)

Tests that the transform_data method gives the correct 10 by 10 matrix.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_apply_template(self)

Tests for the apply template to matrix. A matrix is used which satisfies the constraints of transform_data.
Then, the template is changed, and the applied to the same matrix.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_apply_labeling(self)

Test for the function the users would actually use, for creating full labels from matrix.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_threshold(self)

Tests for when threshold is desired.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_template_init(self)

Checks that other templates are given correctly.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |



