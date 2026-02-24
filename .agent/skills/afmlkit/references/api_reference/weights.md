# API Reference: weights.py

**Language**: Python

**Source**: `label\weights.py`

---

## Functions

### average_uniqueness(timestamps: NDArray[np.int64], event_idxs: NDArray[np.int64], touch_idxs: NDArray[np.int64]) → tuple[NDArray[np.float64], NDArray[np.int16]]

Calculate the uniqueness weights for the overlapping label.
Based on Advances in Financial Machine Learning, Chapter 4. page 61.

:param timestamps: The timestamps in nanoseconds for the close prices series.
:param event_idxs: The indices of the labeled events, e.g. acquired from the cusum filter. (subset of timestamps)
:param touch_idxs: The touch indices for the given events.
:returns: A tuple with two arrays
    - The uniqueness weights [0, 1] for the label.
    - The concurrency array, which indicates how many labels overlap at each timestamp.
:raises ValueError: If timestamps and touch indices are of different lengths.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| timestamps | NDArray[np.int64] | - | - |
| event_idxs | NDArray[np.int64] | - | - |
| touch_idxs | NDArray[np.int64] | - | - |

**Returns**: `tuple[NDArray[np.float64], NDArray[np.int16]]`



### return_attribution(event_idxs: NDArray[np.int64], touch_idxs: NDArray[np.int64], close: NDArray[np.float64], concurrency: NDArray[np.int16], normalize: bool) → NDArray[np.float64]

Assign more weights to samples with higher return attribution.
Advances in Financial Machine Learning, Chapter 4, page 68.

:param event_idxs: Event indices where the label starts.
:param touch_idxs: Touch indices where the label ends.
:param close: Close price array.
:param concurrency: Concurrency array indicating how many labels overlap at each timestamp. From `label_average_uniqueness` function.
:param normalize: If True, normalize the returned weights to sum to the number of events.
:return: NDArray[np.float64]
    An array of return attribution weights for each event.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| event_idxs | NDArray[np.int64] | - | - |
| touch_idxs | NDArray[np.int64] | - | - |
| close | NDArray[np.float64] | - | - |
| concurrency | NDArray[np.int16] | - | - |
| normalize | bool | - | - |

**Returns**: `NDArray[np.float64]`



### time_decay(avg_uniqueness: NDArray[np.float64], last_weight: float) → NDArray[np.float64]

Apply linear time decay based on the average uniqueness weights.
Newest observation assigned with 1.0 and oldest with `last_weight`.
If `last_weight` is negative, the oldest portion (n_events* last_weight) is get erased (assigned with 0.0.)
Advances in Financial Machine Learning, Chapter 4, page 70.

:param avg_uniqueness: The average uniqueness weights for the label from `average_uniqueness` function.
:param last_weight: The weight assigned to the last sample. If 1.0, then there is no decay.
:return: An array of time-decayed weights [0, 1] for each event.
:raises ValueError: The sum of all average uniqueness weights must be greater than 0.
:raises ValueError: If `last_weight` is not in the range [-1, 1].

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| avg_uniqueness | NDArray[np.float64] | - | - |
| last_weight | float | - | - |

**Returns**: `NDArray[np.float64]`



### class_balance_weights(labels: NDArray[np.int8], base_w: NDArray[np.float64]) → Tuple[NDArray[np.int8], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]

Run this function after all other sample weights have been calculated and combined into `base_w`.
Calculate the class balance weights for the given label using the base sample weights.

:param labels: The label (e.g., -1, 0, 1) for the given events.
:param base_w: Base weights for the given label (e.g., avg_uniqueness weights, vertical barrier weights, return attribution, time-decay combined).
       Number of class elements will be calculated as a weighted sum.
:returns: A tuple containing:
    - The identified classes.
    - Corresponding class weights.
    - Number of class elements per label calculated as a sum of sample weights.
    - Final weights array per sample: class weights multiplied by base weights.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| labels | NDArray[np.int8] | - | - |
| base_w | NDArray[np.float64] | - | - |

**Returns**: `Tuple[NDArray[np.int8], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]`


