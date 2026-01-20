# API Reference: test_fracdiff.py

**Language**: Python

**Source**: `tests/test_fracdiff.py`

---

## Classes

### TestFractionalDifferentiation

Test get_weights, get_weights_ffd, frac_diff, and frac_diff_ffd

**Inherits from**: unittest.TestCase

#### Methods

##### setUp(self)

Set the file path for the sample dollar bars data.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_get_weights(self)

get_weights as implemented here matches the code in the book (Snippet 5.1).
We test:
1. if the first element of the weights vector is 1.0
2. The length of the weights vector is same as the requested length

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_get_weights_ffd(self)

get_weights_ffd as implemented here matches the code in the book (Snippet 5.2).
We test:
1. if the first element of the weights vector is 1.0
2. The length of the weights vector is equal to 12

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_frac_diff(self)

Assert that for any positive real number d,
1. Length of the output is the same as the length of the input
2. First element is NaN

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_frac_diff_ffd(self)

Assert that for any positive real number d,
1. Length of the output is the same as the length of the input
2. First element is NaN

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_plot_min_ffd(self)

Assert that the plot for min ffd is correct,

Testing is based on the correlation between the original series (d=0)
and the differentiated series.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |



