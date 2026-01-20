# API Reference: test_cross_validation.py

**Language**: Python

**Source**: `tests/test_cross_validation.py`

---

## Classes

### TestCrossValidation

Test the functionality of the time series cross validation technique

**Inherits from**: unittest.TestCase

#### Methods

##### __init__(self)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### log(self, msg)

Simple method to suppress debugging strings

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| msg | None | - | - |


##### setUp(self)

This is how the observations dataset looks like
2019-01-01 00:00:00   2019-01-01 00:02:00
2019-01-01 00:01:00   2019-01-01 00:03:00
2019-01-01 00:02:00   2019-01-01 00:04:00
2019-01-01 00:03:00   2019-01-01 00:05:00
2019-01-01 00:04:00   2019-01-01 00:06:00
2019-01-01 00:05:00   2019-01-01 00:07:00
2019-01-01 00:06:00   2019-01-01 00:08:00
2019-01-01 00:07:00   2019-01-01 00:09:00
2019-01-01 00:08:00   2019-01-01 00:10:00
2019-01-01 00:09:00   2019-01-01 00:11:00

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_get_train_times_1(self)

Tests the get_train_times method for the case where the train STARTS within test.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_get_train_times_2(self)

Tests the get_train_times method for the case where the train ENDS within test.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_get_train_times_3(self)

Tests the get_train_times method for the case where the train ENVELOPES test.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_purgedkfold_01_exception(self)

Test throw exception when samples_info_sets is not a pd.Series.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_purgedkfold_02_exception(self)

Test exception is raised when passing in a dataset with a different length than the samples_info_sets used in the
constructor.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_purgedkfold_03_simple(self)

Test PurgedKFold class using the ml_get_train_times method. Get the test range from PurgedKFold and then make
sure the train range is exactly the same using the two methods.

This is the test with no embargo.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_purgedkfold_04_embargo(self)

Test PurgedKFold class using the 'embargo' parameter set to pct_points_test which means pct_points_test percent
which also means pct_points_test entries from a total of 100 in total in the dataset.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### _test_ml_cross_val_score__data(self)

Get data structures for next few tests.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_ml_cross_val_score_01_accuracy(self)

Test the ml_cross_val_score function with an artificial dataset.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_ml_cross_val_score_02_neg_log_loss(self)

Test the ml_cross_val_score function with an artificial dataset.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_ml_cross_val_score_03_other_cv_gen(self)

Test the ml_cross_val_score function with an artificial dataset.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_ml_cross_val_score_04_sw(self)

Test the ml_cross_val_score function with an artificial dataset.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |



