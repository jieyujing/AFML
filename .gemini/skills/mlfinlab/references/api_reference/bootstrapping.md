# API Reference: bootstrapping.py

**Language**: Python

**Source**: `sampling/bootstrapping.py`

---

## Functions

### get_ind_matrix(samples_info_sets, price_bars)

Advances in Financial Machine Learning, Snippet 4.3, page 65.

Build an Indicator Matrix

Get indicator matrix. The book implementation uses bar_index as input, however there is no explanation
how to form it. We decided that using triple_barrier_events and price bars by analogy with concurrency
is the best option.

:param samples_info_sets: (pd.Series): Triple barrier events(t1) from labeling.get_events
:param price_bars: (pd.DataFrame): Price bars which were used to form triple barrier events
:return: (np.array) Indicator binary matrix indicating what (price) bars influence the label for each observation

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| samples_info_sets | None | - | - |
| price_bars | None | - | - |

**Returns**: (none)



### get_ind_mat_average_uniqueness(ind_mat)

Advances in Financial Machine Learning, Snippet 4.4. page 65.

Compute Average Uniqueness

Average uniqueness from indicator matrix

:param ind_mat: (np.matrix) Indicator binary matrix
:return: (float) Average uniqueness

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| ind_mat | None | - | - |

**Returns**: (none)



### get_ind_mat_label_uniqueness(ind_mat)

Advances in Financial Machine Learning, An adaption of Snippet 4.4. page 65.

Returns the indicator matrix element uniqueness.

:param ind_mat: (np.matrix) Indicator binary matrix
:return: (np.matrix) Element uniqueness

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| ind_mat | None | - | - |

**Returns**: (none)



### _bootstrap_loop_run(ind_mat, prev_concurrency)

Part of Sequential Bootstrapping for-loop. Using previously accumulated concurrency array, loops through all samples
and generates averages uniqueness array of label based on previously accumulated concurrency

:param ind_mat (np.array): Indicator matrix from get_ind_matrix function
:param prev_concurrency (np.array): Accumulated concurrency from previous iterations of sequential bootstrapping
:return: (np.array): Label average uniqueness based on prev_concurrency

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| ind_mat | None | - | - |
| prev_concurrency | None | - | - |

**Returns**: (none)



### seq_bootstrap(ind_mat, sample_length = None, warmup_samples = None, compare = False, verbose = False, random_state = np.random.RandomState())

Advances in Financial Machine Learning, Snippet 4.5, Snippet 4.6, page 65.

Return Sample from Sequential Bootstrap

Generate a sample via sequential bootstrap.
Note: Moved from pd.DataFrame to np.matrix for performance increase

:param ind_mat: (pd.DataFrame) Indicator matrix from triple barrier events
:param sample_length: (int) Length of bootstrapped sample
:param warmup_samples: (list) List of previously drawn samples
:param compare: (boolean) Flag to print standard bootstrap uniqueness vs sequential bootstrap uniqueness
:param verbose: (boolean) Flag to print updated probabilities on each step
:param random_state: (np.random.RandomState) Random state
:return: (array) Bootstrapped samples indexes

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| ind_mat | None | - | - |
| sample_length | None | None | - |
| warmup_samples | None | None | - |
| compare | None | False | - |
| verbose | None | False | - |
| random_state | None | np.random.RandomState() | - |

**Returns**: (none)


