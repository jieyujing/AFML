# API Reference: time.py

**Language**: Python

**Source**: `feature\core\time.py`

---

## Functions

### time_cues(timestamps: NDArray[np.int64])

:param timestamps : int64 nanoseconds UTC
:returns: tuple of arrays for the block-7 features
          (sin_td, cos_td, dow,, asia, eu, us, sess_x, top_hr)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| timestamps | NDArray[np.int64] | - | - |

**Returns**: (none)


