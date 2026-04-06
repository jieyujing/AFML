"""Parameter grid generator for Primary Model Factory."""

import pandas as pd
from itertools import product


def generate_param_grid(
    cusum_rates: list[float],
    fast_windows: list[int],
    slow_windows: list[int],
    vertical_bars: list[int],
) -> pd.DataFrame:
    """
    Generate valid parameter combination grid.

    :param cusum_rates: List of CUSUM target rates.
    :param fast_windows: List of fast MA windows.
    :param slow_windows: List of slow MA windows.
    :param vertical_bars: List of TBM vertical barrier bar counts.
    :returns: DataFrame with one row per valid combo.
              Columns: combo_id, cusum_rate, fast, slow, vertical_bars.
    """
    combos = []

    for rate, fast, slow, vb in product(cusum_rates, fast_windows, slow_windows, vertical_bars):
        # Constraint: fast < slow
        if fast >= slow:
            continue

        combo_id = f"rate={rate}_fast={fast}_slow={slow}_vb={vb}"
        combos.append({
            'combo_id': combo_id,
            'cusum_rate': rate,
            'fast': fast,
            'slow': slow,
            'vertical_bars': vb,
        })

    df = pd.DataFrame(combos)

    # Ensure columns exist even if empty
    if len(df) == 0:
        df = pd.DataFrame(columns=['combo_id', 'cusum_rate', 'fast', 'slow', 'vertical_bars'])
    else:
        # Sort by rate, fast, slow, vertical_bars
        df = df.sort_values(['cusum_rate', 'fast', 'slow', 'vertical_bars']).reset_index(drop=True)

    return df