# API Reference: volume_classifier.py

**Language**: Python

**Source**: `util/volume_classifier.py`

---

## Functions

### get_bvc_buy_volume(close: pd.Series, volume: pd.Series, window: int = 20) → pd.Series

Calculates the BVC buy volume

:param close: (pd.Series): Close prices
:param volume: (pd.Series): Bar volumes
:param window: (int): Window for std estimation uses in BVC calculation
:return: (pd.Series) BVC buy volume

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| close | pd.Series | - | - |
| volume | pd.Series | - | - |
| window | int | 20 | - |

**Returns**: `pd.Series`


