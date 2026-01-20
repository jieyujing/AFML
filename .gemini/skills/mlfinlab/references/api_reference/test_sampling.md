# API Reference: test_sampling.py

**Language**: Python

**Source**: `tests/test_sampling.py`

---

## Classes

### TestSampling

Test Triple barrier, meta-labeling, dropping rare labels, and daily volatility.

**Inherits from**: unittest.TestCase

#### Methods

##### setUp(self)

Set samples_info_sets (t1), price bars

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_num_concurrent_events(self)

Assert that number of concurent events have are available for all labels and equal to particular values

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_get_av_uniqueness(self)

Assert that average event uniqueness is available for all labels and equals to particular values

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_seq_bootstrap(self)

Test sequential bootstrapping length, indicator matrix length and NaN checks

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_get_ind_mat_av_uniqueness(self)

Tests get_ind_mat_average_uniqueness function using indicator matrix from the book example

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_get_ind_mat_uniqueness(self)

Tests get_ind_mat_uniqueness function using indicator matrix from the book example

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_bootstrap_loop_run(self)

Test one loop iteration of Sequential Bootstrapping

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_value_error_raise(self)

Test seq_bootstrap and ind_matrix functions for raising ValueError on nan values

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |




## Functions

### book_ind_mat_implementation(bar_index, label_endtime)

Book implementation of get_ind_matrix function

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| bar_index | None | - | - |
| label_endtime | None | - | - |

**Returns**: (none)


