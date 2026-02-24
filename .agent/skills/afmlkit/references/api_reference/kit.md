# API Reference: kit.py

**Language**: Python

**Source**: `label\kit.py`

---

## Classes

### TBMLabel

Implements the Triple Barrier Method (TBM) for labeling financial events, as described by Marcos Lopez de Prado.
This method assigns labels to events based on whether the price touches an upper barrier (take-profit), lower barrier (stop-loss),
or a vertical time barrier first. It supports both side labeling and meta-labeling modes.

The Triple Barrier Method is a technique for labeling outcomes in financial machine learning, particularly useful for
creating supervised learning datasets from time-series data. It helps mitigate issues like overfitting and improves
the informativeness of labels by considering profitability thresholds and time horizons.

For a set of events (e.g., trading signals or cusum events), the method constructs three barriers around each event's
starting price:

- **Upper horizontal barrier**: Take-profit level, computed as starting price plus (target return * upper multiplier).
- **Lower horizontal barrier**: Stop-loss level, computed as starting price minus (target return * lower multiplier).
- **Vertical barrier**: A time-based barrier after a specified timedelta.

The label is determined by which barrier is touched first by the price path:

- +1 if upper barrier is touched first (profitable).
- -1 if lower barrier is touched first (loss).
- 0 if vertical barrier is touched first (timeout), or adjustable based on meta-labeling.

In meta-labeling mode (``is_meta=True``), the method incorporates predictions from a primary model (via the 'side' column).
Labels are assigned only if the primary model's direction aligns with the barrier outcome, enabling meta-models to learn
when to trust the primary model.

Mathematically, for an event at time :math:`t` with starting price :math:`p_t`, target return :math:`r_t` (e.g., volatility estimate),
and horizontal multipliers :math:`(m_{low}, m_{up})`:

.. math::
    \text{Upper barrier} = p_t \cdot (1 + r_t \cdot m_{up})

    \text{Lower barrier} = p_t \cdot (1 - r_t \cdot m_{low})

    \text{Vertical barrier} = t + \Delta t

The label :math:`l` for the event is:

.. math::
    l = \begin{cases}
    1, & \text{if upper barrier touched first} \\
    -1, & \text{if lower barrier touched first} \\
    0, & \text{if vertical barrier touched first}
    \end{cases}

.. important::
    In this implementation, we are constructing binary labels: either +1 or -1 for side prediction as recommended in
    Advances in Financial Machine Learning. We introduce "vertical_touch_weights" to decrease the weights of misleading labels associated with a vertical barrier touch.
    Consider the following scenario: vertical barrier is hit slightly **above**/below the initial price resulting in **1**/-1 label,
    but the price path was very close to the **lower**/upper barrier (almost hit it). If the ML model predicted **-1**/1 for this event, we don't want to heavily punish it.

In meta-labeling, the label is modulated by the primary side :math:`s \in \{-1, 1\}`:

.. math::
    l_{meta} = \begin{cases}
    1, & \text{if } (s = 1 \land l = 1) \lor (s = -1 \land l = -1) \\
    0, & \text{otherwise}
    \end{cases}

.. note::
    To disable a horizontal barrier, set its multiplier to :math:`+\infty` or :math:`-\infty`. For the vertical barrier,
    use a very large timedelta (e.g., 1000 years) to effectively disable it.

.. note::
    This implementation supports computation of sample weights via the related :class:`SampleWeights` class.
    After labeling, use :meth:`compute_weights` to calculate information-driven weights, including:

    - **Label concurrency**: Measures overlap of event durations.
    - **Return attribution**: Attributes returns to overlapping events proportionally to their uniqueness.

    These can be combined with time decay and class balancing for final sample weights in model training using :meth:`SampleWeights.compute_final_weights`.

.. _`Advances in Financial Machine Learning`: https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086

Args:
    features (pd.DataFrame): The events dataframe containing the return target column and optionally event indices ("event_idx" column) and features. If not provided, event indices will be computed based on timestamps.
    target_ret_col (str): The name of the target return column in the ``features`` dataframe. Typically a volatility estimator output. This is used to scale the horizontal barriers. Should be in log-return space.
    min_ret (float): Minimum required return threshold. Events where the absolute target return (scaled by max horizontal multiplier) is below this threshold will be dropped.
    horizontal_barriers (tuple[float, float]): Bottom and top (stop-loss/take-profit) horizontal barrier multipliers. The target return is multiplied by these to determine barrier widths. Use -inf/+inf to disable.
    vertical_barrier (pd.Timedelta): The temporal barrier duration. Set to a large value (e.g., pd.Timedelta(days=365*1000)) to disable.
    min_close_time (pd.Timedelta, optional): Prevents premature event closure before this minimum time. Default: pd.Timedelta(seconds=1).
    is_meta (bool, optional): Enable meta-labeling mode. If True, ``features`` must contain a 'side' column with primary model predictions (-1, 0, 1). Default: False.

Raises:
    ValueError: If input validations fail, such as missing columns, invalid types, or empty data after filtering.

See Also:
    :class:`SampleWeights`: For computing the final sample weights combining and normalizing average uniqueness, return attribution, time decay, and class balancing.

**Inherits from**: (none)

#### Methods

##### __init__(self, features: pd.DataFrame, target_ret_col: str, min_ret: float, horizontal_barriers: tuple[float, float], vertical_barrier: pd.Timedelta, min_close_time: pd.Timedelta = pd.Timedelta(seconds=1), is_meta: bool = False)

Triple barrier labeling method

:param features: The events dataframe the return target column and optionally containing the event indices ("event_idx" column) and features. If not it will be computed based on timestamps.
:param target_ret_col: The name of the target return column in the `features` dataframe.
    Typically, a volatility estimator output.
    This will be used to determine the horizontal barriers.
    Should be in log-return space.
:param min_ret: Minimum required return threshold.
    Where `target_col` is below this threshold events will be dropped.
:param horizontal_barriers: Bottom and Top (SL/TP) horizontal barrier multipliers.
    The return target will be multiplied by these multipliers. Determines the width of the horizontal barriers.
    If you want to disable the barriers, set it to -np.inf or +np.inf, respectively.
:param vertical_barrier: The temporal barrier as timedelta. Set it to a large value to disable the vertical barrier (eg. 1000 years)
:param min_close_time: This prevents closing the event prematurely before the minimum close time is reached. Default is 1 second.
:param is_meta: Side or meta labeling.
    If `True` `features` must contain `side` column containing the predictions of the primary model.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| features | pd.DataFrame | - | - |
| target_ret_col | str | - | - |
| min_ret | float | - | - |
| horizontal_barriers | tuple[float, float] | - | - |
| vertical_barrier | pd.Timedelta | - | - |
| min_close_time | pd.Timedelta | pd.Timedelta(seconds=1) | - |
| is_meta | bool | False | - |


##### _preprocess_features(x: pd.DataFrame, target_ret_col: str, min_ret: float, horizontal_barriers: tuple[float, float]) → pd.DataFrame

**Decorators**: `@staticmethod`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| x | pd.DataFrame | - | - |
| target_ret_col | str | - | - |
| min_ret | float | - | - |
| horizontal_barriers | tuple[float, float] | - | - |

**Returns**: `pd.DataFrame`


##### event_count(self) → int

Get the number of events in the features DataFrame.
:return: The number of events.

**Decorators**: `@property`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |

**Returns**: `int`


##### first_event_timestamp(self) → pd.Timestamp | None

Get the timestamp of the first event.
:return: The timestamp of the first event.

**Decorators**: `@property`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |

**Returns**: `pd.Timestamp | None`


##### last_event_timestamp(self) → pd.Timestamp | None

Get the timestamp of the last event.
:return: The timestamp of the last event.

**Decorators**: `@property`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |

**Returns**: `pd.Timestamp | None`


##### event_range(self) → str

Get the range of event timestamps.
:return: A string containing the first and last event timestamps.

**Decorators**: `@property`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |

**Returns**: `str`


##### features(self) → pd.DataFrame

Get the features corresponding the generated labels.
I might be a subset of the original features DataFrame due to TBM evaluation window.
:return: The features DataFrame.

**Decorators**: `@property`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |

**Returns**: `pd.DataFrame`


##### target_returns(self) → pd.Series

Get the target returns for the events.
:return: A pandas Series containing the target returns.

**Decorators**: `@property`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |

**Returns**: `pd.Series`


##### labels(self) → pd.Series

Get the labels for the events.
:return: A pandas Series containing the labels.

**Decorators**: `@property`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |

**Returns**: `pd.Series`


##### event_returns(self) → pd.Series

Get the log returns associated with each event.
:return: A pandas Series containing the log returns.

**Decorators**: `@property`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |

**Returns**: `pd.Series`


##### full_output(self) → pd.DataFrame

Get the full output DataFrame containing labels, event indices, touch indices, returns, and weights.
:return: A pandas DataFrame containing the full output.

**Decorators**: `@property`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |

**Returns**: `pd.DataFrame`


##### _drop_trailing_events(self, trades: TradesData) → pd.DataFrame

We should drop the trailing events which cannot be evaluated on the full temporal window.
:param trades: Raw trades data the events will be evaluated.
:return: Trimmed features DataFrame with events that are within the base series.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| trades | TradesData | - | - |

**Returns**: `pd.DataFrame`


##### compute_labels(self, trades: TradesData) → tuple[pd.DataFrame, pd.DataFrame]

Compute the labels for the events using the triple barrier method.

:param trades: The raw trades data the events will be evaluated
:return: A tuple containing:
    - The features DataFrame with the event indices and other features.
    - A dataframe containing labels, event indices, touch indices, returns, and weights.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| trades | TradesData | - | - |

**Returns**: `tuple[pd.DataFrame, pd.DataFrame]`


##### compute_weights(self, trades: TradesData, normalized: bool = False) → pd.DataFrame

Computes the sample average uniqueness and return attribution.
:param trades: Same Raw trades data passed to `compute_labels()`.
:param normalized: Whether to normalize the weights.
:return: DataFrame containing the sample average uniqueness and return attribution.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| trades | TradesData | - | - |
| normalized | bool | False | - |

**Returns**: `pd.DataFrame`




### SampleWeights

A wrapper class for time decay and class balance weights calculation.
These weights should be run on the training window part of the full dataset.

**Inherits from**: (none)

#### Methods

##### compute_info_weights(trades: TradesData, labels: pd.DataFrame, normalize: bool = False) → pd.DataFrame

Computes the average uniqueness and (non-normalized) return attribution for the events.

:param trades: The raw trades on which the events are evaluated
:param labels: Labels dataframe containing event indices and touch indices (output of `compute_labels` method).
:param normalize: Whether to normalize the returned weights.
:return:  A pandas DataFrame containing the average uniqueness and return attribution and vertical touch weights.

**Decorators**: `@staticmethod`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| trades | TradesData | - | - |
| labels | pd.DataFrame | - | - |
| normalize | bool | False | - |

**Returns**: `pd.DataFrame`


##### compute_final_weights(avg_uniqueness: pd.Series, time_decay_intercept: float = 1.0, return_attribution: pd.Series = None, vertical_touch_weights: pd.Series = None, labels: pd.Series = None) → pd.DataFrame

Compute the time decay and class balance weights based on the average uniqueness and return attribution.
Normalizes return attribution to sum up to event count.

:param avg_uniqueness: Average uniqueness weights for the events.
:param return_attribution: Provide unnormalized return attribution if use this as info weights instead of average uniqueness.
:param vertical_touch_weights: Provide vertical touch weights if you want to apply them to the final weights.
:param time_decay_intercept: The intercept for the time decay function. 1.0 means no decay, 0.0 means full decay. Negative values will erase the oldest portion of the weights.
:param labels: Provide labels if you want to apply class balancing to the final weights.
:return: A pandas Dataframe containing the weight parts and the combined weights.

**Decorators**: `@staticmethod`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| avg_uniqueness | pd.Series | - | - |
| time_decay_intercept | float | 1.0 | - |
| return_attribution | pd.Series | None | - |
| vertical_touch_weights | pd.Series | None | - |
| labels | pd.Series | None | - |

**Returns**: `pd.DataFrame`



