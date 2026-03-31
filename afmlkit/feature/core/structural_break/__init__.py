"""
Structural break features for AFMLKit.

Detects regime changes, bubbles, and trend mutations in financial time series.

Features:
- ADF: Augmented Dickey-Fuller base test
- SADF: Supremum ADF for bubble detection
- QADF: Quantile ADF for noise reduction
- CADF: Conditional ADF for bubble strength quantification
- SMT: Sub/Super-Martingale tests for trend detection
- CUSUM: Chu-Stinchcombe-White CUSUM test

Reference: AFML Chapter 17
"""

# ADF base
from afmlkit.feature.core.structural_break.adf import (
    adf_test,
    adf_test_rolling,
    adf_test_full,
    schwert_maxlag,
)

# SADF
from afmlkit.feature.core.structural_break.sadf import (
    sadf_test,
    SADFTest,
)

# QADF
from afmlkit.feature.core.structural_break.qadf import (
    qadf_test,
    QADFTest,
)

# CADF
from afmlkit.feature.core.structural_break.cadf import (
    cadf_test,
    CADFTest,
)

# SMT (Sub/Super-Martingale)
from afmlkit.feature.core.structural_break.smt import (
    sub_martingale_test,
    super_martingale_test,
    martingale_test,
    SubMartingaleTest,
    SuperMartingaleTest,
    MartingaleTest,
)

# CUSUM
from afmlkit.feature.core.structural_break.cusum import (
    cusum_test_developing,
    cusum_test_last,
    cusum_test_rolling,
)

__all__ = [
    # ADF
    'adf_test',
    'adf_test_rolling',
    'adf_test_full',
    'schwert_maxlag',
    # SADF
    'sadf_test',
    'SADFTest',
    # QADF
    'qadf_test',
    'QADFTest',
    # CADF
    'cadf_test',
    'CADFTest',
    # SMT
    'sub_martingale_test',
    'super_martingale_test',
    'martingale_test',
    'SubMartingaleTest',
    'SuperMartingaleTest',
    'MartingaleTest',
    # CUSUM
    'cusum_test_developing',
    'cusum_test_last',
    'cusum_test_rolling',
]
