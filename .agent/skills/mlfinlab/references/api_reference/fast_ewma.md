# API Reference: fast_ewma.py

**Language**: Python

**Source**: `util/fast_ewma.py`

---

## Functions

### ewma(arr_in, window)

Exponentially weighted moving average specified by a decay ``window`` to provide better adjustments for
small windows via:
    y[t] = (x[t] + (1-a)*x[t-1] + (1-a)^2*x[t-2] + ... + (1-a)^n*x[t-n]) /
           (1 + (1-a) + (1-a)^2 + ... + (1-a)^n).

:param arr_in: (np.ndarray), (float64) A single dimensional numpy array
:param window: (int64) The decay window, or 'span'
:return: (np.ndarray) The EWMA vector, same length / shape as ``arr_in``

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| arr_in | None | - | - |
| window | None | - | - |

**Returns**: (none)


