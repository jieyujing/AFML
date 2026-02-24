# API Reference: reversion.py

**Language**: Python

**Source**: `feature\core\reversion.py`

---

## Functions

### vwap_distance(close: NDArray[np.float64], volume: NDArray[np.float64], n_periods: int, is_log: bool) → NDArray[np.float64]

Calculate the distance of the current price from the VWAP (Volume Weighted Average Price).
The VWAP is calculated over a specified number of periods, and the distance is expressed as a percentage.
:param close: close price series
:param volume: corresponding volume series
:param n_periods: number of periods to calculate VWAP
:param is_log: if True, calculate log distance
:return: array of distances from VWAP

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| close | NDArray[np.float64] | - | - |
| volume | NDArray[np.float64] | - | - |
| n_periods | int | - | - |
| is_log | bool | - | - |

**Returns**: `NDArray[np.float64]`


