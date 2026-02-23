## Context

Current implementation of `DollarBarKit` uses a fixed global dollar volume threshold. We need a way to build dollar bars where the threshold isn't static but rather dynamically tracking the Exponentially Weighted Moving Average (EWMA) of daily dollar volume to adapt to long-term market trends.

## Goals
- Dynamically scale the dollar volume threshold to adapt to market activity over time using an EWMA filter.
- Support parameter sweeping (e.g., target bars per day: 4, 6, 10, 20, 50).
- Automatically test distributions for normality (using `scipy.stats.jarque_bera`) and autocorrelation.
- Pick the frequency mapping corresponding to the best (lowest) JB + low ACF score.
- Output visually using `matplotlib` or `plotly` aligned with `afmlkit` style.

## Non-Goals
- We are not redesigning the underlying `Numba` C-extensions entirely. Rather we will create a higher-level script or class that applies the dynamic threshold generation across chunks/days or provides a tick-by-tick array. If we extend Numba, it will be strictly bounded to `_dynamic_dollar_bar_indexer` supporting array thresholds, or use python-level chunking. We'll implement a `DynamicDollarBarKit` wrapper bridging this.

## Decisions

### Decision 1: Generating the dynamic array of thresholds
We will extract daily aggregate dollar volume mapping to the timestamps.
We apply an EWMA over this daily series. For a given target daily frequency \( f \), the dynamic daily threshold is:
`Threshold(t) = EWMA_Daily_Dollar_Volume(t) / f`
For each tick in the sequence, we map its timestamp to the active threshold of its corresponding day.

### Decision 2: Modifying logic.py
We will add `_dynamic_dollar_bar_indexer(prices, volumes, thresholds_array)` in `afmlkit.bar.logic` modeled after the original but utilizing an array of thresholds `threshold[i]` dynamically updated per tick.

### Decision 3: Creating the testing script
A separate script `scripts/dynamic_dollar_bars.py` will read parquet/HDF5 trades, run the multiple `DynamicDollarBarKit` builders, evaluate JB, find the optimal target frequency \( f_{opt} \), plot properties and the final bar series.
