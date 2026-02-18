# API Reference: third_generation.py

**Language**: Python

**Source**: `microstructural_features/third_generation.py`

---

## Functions

### get_vpin(volume: pd.Series, buy_volume: pd.Series, window: int = 1) → pd.Series

Advances in Financial Machine Learning, p. 292-293.

Get Volume-Synchronized Probability of Informed Trading (VPIN) from bars

:param volume: (pd.Series) Bar volume
:param buy_volume: (pd.Series) Bar volume classified as buy (either tick rule, BVC or aggressor side methods applied)
:param window: (int) Estimation window
:return: (pd.Series) VPIN series

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| volume | pd.Series | - | - |
| buy_volume | pd.Series | - | - |
| window | int | 1 | - |

**Returns**: `pd.Series`


