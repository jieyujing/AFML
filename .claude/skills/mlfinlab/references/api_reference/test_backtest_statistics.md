# API Reference: test_backtest_statistics.py

**Language**: Python

**Source**: `tests/test_backtest_statistics.py`

---

## Classes

### TestBacktestStatistics

Test following functions in statistocs.py:
- timing_of_flattening_and_flips
- average_holding_period
- bets_concentration
- all_bets_concentration
- compute_drawdown_and_time_under_water
- sharpe_ratio
- information ratio
- probabilistic_sharpe_ratio
- deflated_sharpe_ratio
- minimum_track_record_length

**Inherits from**: unittest.TestCase

#### Methods

##### setUp(self)

Set the data for tests.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_timing_of_flattening_and_flips(self)

Check that moments of flips and flattenings are picked correctly and
that last is added

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_average_holding_period(self)

Check average holding period calculation

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_bets_concentration(self)

Check if concentration is balanced and correctly calculated

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_all_bets_concentration(self)

Check if concentration is nan when not enough observations, also values
testing

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_drawdown_and_time_under_water(self)

Check if drawdowns and time under water calculated correctly for
dollar and non-dollar test sets.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_sharpe_ratio(self)

Check if Sharpe ratio is calculated right

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_information_ratio(self)

Check if Information ratio is calculated right

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_probabilistic_sharpe_ratio(self)

Check probabilistic Sharpe ratio using numerical example

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_deflated_sharpe_ratio(self)

Check deflated Sharpe ratio using numerical example

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_minimum_track_record_length(self)

Check deflated Sharpe ratio using numerical example

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |



