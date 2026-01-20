# API Reference: test_labeling_vs_benchmark.py

**Language**: Python

**Source**: `tests/test_labeling_vs_benchmark.py`

---

## Classes

### TestReturnOverBenchmark

Tests regarding the labeling returns over benchmark method.

**Inherits from**: unittest.TestCase

#### Methods

##### setUp(self)

Set the file path for the sample dollar bars data.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_basic(self)

Tests for the basic case where the benchmark is a constant.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_given_benchmark(self)

Tests comparing value to a dynamic benchmark.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_resample(self)

Tests for when resampling is used.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_exception(self)

Verifies that the exception is given when there is a mismatch between prices.index and benchmark.index.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |



