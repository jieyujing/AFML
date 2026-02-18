# API Reference: test_online_portfolio_selection.py

**Language**: Python

**Source**: `tests/test_online_portfolio_selection.py`

---

## Classes

### TestOLPS

Tests different functions of the OLPS class.

**Inherits from**: TestCase

#### Methods

##### setUp(self)

Sets the file path for the tick data csv.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_olps_solution(self)

Test the calculation of OLPS weights.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_olps_weight(self)

Tests that the user inputted weights have matching dimensions as the data's dimensions
and ValueError if the user inputted weights do not sum to one.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_olps_incorrect_data(self)

Tests ValueError if the user inputted data is not a dataframe.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_olps_index_error(self)

Tests ValueError if the passing dataframe is not indexed by date.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_user_weight(self)

Tests that users can input their own weights for OLPS.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_uniform_weight(self)

Tests that uniform weights return equal allocation of weights.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_normalize(self)

Tests that weights sum to 1.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_simplex_projection(self)

Tests edge cases where the inputted weights already satisfy the simplex requirements.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_progress_bar(self)

Tests that verbose=True prints out progress bar.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_simplex_all_negatives(self)

Tests case where negative weights have to be projected onto the simplex.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_null_zero_date(self)

Tests ValueError for data with values of null or zero.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |



