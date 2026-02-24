# API Reference: correlation.py

**Language**: Python

**Source**: `feature\core\correlation.py`

---

## Functions

### rolling_price_volume_correlation(price: NDArray[np.float64], volume: NDArray[np.float64], window: int) → NDArray[np.float64]

Calculate the rolling Pearson correlation coefficient between price returns and volume.

:param price: Array of price values
:param volume: Array of volume values
:param window: Window size for rolling correlation
:return: Array of correlation coefficients

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| price | NDArray[np.float64] | - | - |
| volume | NDArray[np.float64] | - | - |
| window | int | - | - |

**Returns**: `NDArray[np.float64]`


