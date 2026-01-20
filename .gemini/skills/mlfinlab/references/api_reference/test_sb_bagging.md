# API Reference: test_sb_bagging.py

**Language**: Python

**Source**: `tests/test_sb_bagging.py`

---

## Classes

### TestSequentiallyBootstrappedBagging

Test SequentiallyBootstrapped Bagging classifiers

**Inherits from**: unittest.TestCase

#### Methods

##### setUp(self)

Set the file path for the sample dollar bars data and get triple barrier events, generate features

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_sb_bagging_not_tree_base_estimator(self)

Test SB Bagging with non-tree base estimator (KNN)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_sb_bagging_non_sample_weights_with_verbose(self)

Test SB Bagging with classifier which doesn't support sample_weights with verbose > 1

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_sb_bagging_with_max_features(self)

Test SB Bagging with base_estimator bootstrap = True, float max_features, max_features bootstrap = True
:return:

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_sb_bagging_float_max_samples_warm_start_true(self)

Test SB Bagging with warm start = True and float max_samples

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_value_error_raise(self)

Test various values error raise

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_sb_classifier(self)

Test Sequentially Bootstrapped Bagging Classifier. Here we compare oos/oob scores to sklearn's bagging oos scores,
test oos predictions values

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_sb_regressor(self)

Test Sequentially Bootstrapped Bagging Regressor

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |




## Functions

### _generate_label_with_prob(x, prob, random_state = np.random.RandomState(1))

Generates true label value with some probability(prob)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| x | None | - | - |
| prob | None | - | - |
| random_state | None | np.random.RandomState(1) | - |

**Returns**: (none)



### _get_synthetic_samples(ind_mat, good_samples_thresh, bad_samples_thresh)

Get samples with uniqueness either > good_samples_thresh or uniqueness < bad_samples_thresh

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| ind_mat | None | - | - |
| good_samples_thresh | None | - | - |
| bad_samples_thresh | None | - | - |

**Returns**: (none)


