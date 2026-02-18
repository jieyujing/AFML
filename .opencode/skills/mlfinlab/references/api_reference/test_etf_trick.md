# API Reference: test_etf_trick.py

**Language**: Python

**Source**: `tests/test_etf_trick.py`

---

## Classes

### TestETFTrick

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


##### test_etf_trick_costs_defined(self)

Tests in-memory and csv ETF trick implementation, when costs_df is defined

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_etf_trick_rates_not_defined(self)

Tests in-memory and csv ETF trick implementation, when costs_df is not defined (should be set trivial)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_input_exceptions(self)

Tests input data frames internal checks

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |



