# API Reference: test_labels.py

**Language**: Python

**Source**: `tests/test_labels.py`

---

## Classes

### TestChapter3

Test Triple barrier, meta-labeling, dropping rare labels, and daily volatility.

**Inherits from**: unittest.TestCase

#### Methods

##### setUp(self)

Set the file path for the sample dollar bars data.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_daily_volatility(self)

Daily vol as implemented here matches the code in the book.
Although I have reservations, example: no minimum value is set in the EWM.
Thus it returns values for volatility before there are even enough data points.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_vertical_barriers(self)

Assert that the vertical barrier returns the timestamp x amount of days after the event.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_triple_barrier_events(self)

Assert that the different version of triple barrier labeling match our expected output.
Assert that trgts are the same for all 3 methods.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_triple_barrier_labeling(self)

Assert that meta labeling as well as standard labeling works. Also check that if a vertical barrier is
reached, then a 0 class label is assigned (in the case of standard labeling).

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_pt_sl_levels_triple_barrier_events(self)

Previously a bug was introduced by not multiplying the target by the profit taking / stop loss multiple. This
meant that the get_bins function would not return the correct label. Example: if take profit was set to 1000,
it would ignore this multiple and use only the target value. This meant that if we set a very large pt value
(so high that it would never be hit before the vertical barrier is reached), it would ignore the multiple and
only use the target value (it would signal that price reached the pt barrier). This meant that vertical barriers
were incorrectly labeled.

This also meant that irrespective of the pt_sl levels set, the labels would always be the same.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_drop_labels(self)

Assert that drop_labels removes rare class labels.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |



