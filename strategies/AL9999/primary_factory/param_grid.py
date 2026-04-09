"""Parameter grid generators for Primary Model Factory."""

from itertools import product

import pandas as pd


def generate_lightweight_param_grid(
    cusum_rates: list[float],
    fast_windows: list[int],
    slow_windows: list[int],
) -> pd.DataFrame:
    """
    Generate valid lightweight parameter combinations.

    Lightweight search only explores event density plus DMA windows.
    Vertical-bar expansion is deferred to deep scoring.

    :param cusum_rates: List of CUSUM target rates.
    :param fast_windows: List of fast MA windows.
    :param slow_windows: List of slow MA windows.
    :returns: DataFrame with columns [combo_id, cusum_rate, fast, slow].
    """
    combos = []

    for rate, fast, slow in product(cusum_rates, fast_windows, slow_windows):
        if fast >= slow:
            continue

        combos.append({
            "combo_id": f"rate={rate}_fast={fast}_slow={slow}",
            "cusum_rate": rate,
            "fast": fast,
            "slow": slow,
        })

    df = pd.DataFrame(combos)
    if len(df) == 0:
        return pd.DataFrame(columns=["combo_id", "cusum_rate", "fast", "slow"])

    return df.sort_values(["cusum_rate", "fast", "slow"]).reset_index(drop=True)


def expand_deep_param_grid(
    lightweight_combos: pd.DataFrame,
    vertical_bars: list[int],
) -> pd.DataFrame:
    """
    Expand lightweight winners into deep-scoring combinations.

    :param lightweight_combos: DataFrame with columns [combo_id, cusum_rate, fast, slow].
    :param vertical_bars: TBM vertical barrier candidates.
    :returns: DataFrame with columns
              [combo_id, base_combo_id, cusum_rate, fast, slow, vertical_bars].
    """
    combos = []

    for _, row in lightweight_combos.iterrows():
        for vb in vertical_bars:
            combos.append({
                "combo_id": f"{row['combo_id']}_vb={int(vb)}",
                "base_combo_id": row["combo_id"],
                "cusum_rate": row["cusum_rate"],
                "fast": int(row["fast"]),
                "slow": int(row["slow"]),
                "vertical_bars": int(vb),
            })

    df = pd.DataFrame(combos)
    if len(df) == 0:
        return pd.DataFrame(
            columns=["combo_id", "base_combo_id", "cusum_rate", "fast", "slow", "vertical_bars"]
        )

    return df.sort_values(
        ["cusum_rate", "fast", "slow", "vertical_bars"]
    ).reset_index(drop=True)


def generate_param_grid(
    cusum_rates: list[float],
    fast_windows: list[int],
    slow_windows: list[int],
    vertical_bars: list[int],
) -> pd.DataFrame:
    """
    Backward-compatible full grid generator.

    :param cusum_rates: List of CUSUM target rates.
    :param fast_windows: List of fast MA windows.
    :param slow_windows: List of slow MA windows.
    :param vertical_bars: List of TBM vertical barrier bar counts.
    :returns: Full expanded DataFrame.
    """
    lightweight = generate_lightweight_param_grid(
        cusum_rates=cusum_rates,
        fast_windows=fast_windows,
        slow_windows=slow_windows,
    )
    return expand_deep_param_grid(
        lightweight_combos=lightweight,
        vertical_bars=vertical_bars,
    )
