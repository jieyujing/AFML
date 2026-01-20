# API Reference: test_time_data_structures.py

**Language**: Python

**Source**: `tests/test_time_data_structures.py`

---

## Classes

### TestTimeDataStructures

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


##### test_day_bars(self)

Tests the seconds bars implementation.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_hour_bars(self)

Tests the seconds bars implementation.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_minute_bars(self)

Tests the minute bars implementation.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_second_bars(self)

Tests the seconds bars implementation.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_wrong_input_value_error_raise(self)

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



