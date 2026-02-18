# API Reference: test_imbalance_data_structures.py

**Language**: Python

**Source**: `tests/test_imbalance_data_structures.py`

---

## Classes

### TestDataStructures

Test the various financial data structures:
1. Imbalance Dollar bars
2. Imbalance Volume bars
3. Imbalance Tick bars

**Inherits from**: unittest.TestCase

#### Methods

##### setUp(self)

Set the file path for the tick data csv

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_ema_imbalance_dollar_bars(self)

Tests the EMA imbalance dollar bars implementation.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_ema_imbalance_volume_bars(self)

Tests the EMA imbalance volume bars implementation.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_ema_imbalance_tick_bars(self)

Tests the EMA imbalance tick bars implementation.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_ema_imb_dollar_bars_with_constraints(self)

Test the EMA Dollar Imbalance bars with expected number of ticks max and min constraints

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_const_imbalance_dollar_bars(self)

Tests the Const imbalance dollar bars implementation.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_const_imbalance_volume_bars(self)

Tests the Const imbalance volume bars implementation.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_const_imbalance_tick_bars(self)

Tests the Const imbalance tick bars implementation.

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


##### test_wrong_imbalance_passed(self)

Tests ValueError raise when wrong imbalance was passed

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |



