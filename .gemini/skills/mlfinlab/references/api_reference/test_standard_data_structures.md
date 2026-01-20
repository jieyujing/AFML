# API Reference: test_standard_data_structures.py

**Language**: Python

**Source**: `tests/test_standard_data_structures.py`

---

## Classes

### TestDataStructures

Test the various financial data structures:
1. Dollar bars
2. Volume bars
3. Tick bars

**Inherits from**: unittest.TestCase

#### Methods

##### setUp(self)

Set the file path for the tick data csv

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_dollar_bars(self)

Tests the dollar bars implementation.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_volume_bars(self)

Tests the volume bars implementation.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_tick_bars(self)

Test the tick bars implementation.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_multiple_csv_file_input(self)

Tests that bars generated for multiple csv files and Pandas Data Frame yield the same result

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_df_as_batch_run_input(self)

Tests that bars generated for csv file and Pandas Data Frame yield the same result

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_list_as_run_input(self)

Tests that data generated with csv file and list yield the same result

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_wrong_batch_input_value_error_raise(self)

Tests ValueError raise when neither pd.DataFrame nor path to csv file are passed to function call

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_wrong_run_input_value_error_raise(self)

Tests ValueError raise when neither pd.DataFrame nor path to csv file are passed to function call

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_csv_format(self)

Asserts that the csv data being passed is of the correct format.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |



