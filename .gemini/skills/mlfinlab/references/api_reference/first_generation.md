# API Reference: first_generation.py

**Language**: Python

**Source**: `microstructural_features/first_generation.py`

---

## Functions

### get_roll_measure(close_prices: pd.Series, window: int = 20) → pd.Series

Advances in Financial Machine Learning, page 282.

Get Roll Measure

Roll Measure gives the estimate of effective bid-ask spread
without using quote-data.

:param close_prices: (pd.Series) Close prices
:param window: (int) Estimation window
:return: (pd.Series) Roll measure

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| close_prices | pd.Series | - | - |
| window | int | 20 | - |

**Returns**: `pd.Series`



### get_roll_impact(close_prices: pd.Series, dollar_volume: pd.Series, window: int = 20) → pd.Series

Get Roll Impact.

Derivate from Roll Measure which takes into account dollar volume traded.

:param close_prices: (pd.Series) Close prices
:param dollar_volume: (pd.Series) Dollar volume series
:param window: (int) Estimation window
:return: (pd.Series) Roll impact

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| close_prices | pd.Series | - | - |
| dollar_volume | pd.Series | - | - |
| window | int | 20 | - |

**Returns**: `pd.Series`



### _get_beta(high: pd.Series, low: pd.Series, window: int) → pd.Series

Advances in Financial Machine Learning, Snippet 19.1, page 285.

Get beta estimate from Corwin-Schultz algorithm

:param high: (pd.Series) High prices
:param low: (pd.Series) Low prices
:param window: (int) Estimation window
:return: (pd.Series) Beta estimates

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| high | pd.Series | - | - |
| low | pd.Series | - | - |
| window | int | - | - |

**Returns**: `pd.Series`



### _get_gamma(high: pd.Series, low: pd.Series) → pd.Series

Advances in Financial Machine Learning, Snippet 19.1, page 285.

Get gamma estimate from Corwin-Schultz algorithm.

:param high: (pd.Series) High prices
:param low: (pd.Series) Low prices
:return: (pd.Series) Gamma estimates

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| high | pd.Series | - | - |
| low | pd.Series | - | - |

**Returns**: `pd.Series`



### _get_alpha(beta: pd.Series, gamma: pd.Series) → pd.Series

Advances in Financial Machine Learning, Snippet 19.1, page 285.

Get alpha from Corwin-Schultz algorithm.

:param beta: (pd.Series) Beta estimates
:param gamma: (pd.Series) Gamma estimates
:return: (pd.Series) Alphas

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| beta | pd.Series | - | - |
| gamma | pd.Series | - | - |

**Returns**: `pd.Series`



### get_corwin_schultz_estimator(high: pd.Series, low: pd.Series, window: int = 20) → pd.Series

Advances in Financial Machine Learning, Snippet 19.1, page 285.

Get Corwin-Schultz spread estimator using high-low prices

:param high: (pd.Series) High prices
:param low: (pd.Series) Low prices
:param window: (int) Estimation window
:return: (pd.Series) Corwin-Schultz spread estimators

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| high | pd.Series | - | - |
| low | pd.Series | - | - |
| window | int | 20 | - |

**Returns**: `pd.Series`



### get_bekker_parkinson_vol(high: pd.Series, low: pd.Series, window: int = 20) → pd.Series

Advances in Financial Machine Learning, Snippet 19.2, page 286.

Get Bekker-Parkinson volatility from gamma and beta in Corwin-Schultz algorithm.

:param high: (pd.Series) High prices
:param low: (pd.Series) Low prices
:param window: (int) Estimation window
:return: (pd.Series) Bekker-Parkinson volatility estimates

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| high | pd.Series | - | - |
| low | pd.Series | - | - |
| window | int | 20 | - |

**Returns**: `pd.Series`


