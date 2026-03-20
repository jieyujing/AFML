"""
Structural break detection features for AFMLKit.

This module implements various structural break tests based on the
Advances in Financial Machine Learning methodology:

- ADF (Augmented Dickey-Fuller) test
- SADF (Supremum ADF) test for bubble detection
- CUSUM structural break detection

Reference: AFML Chapter 17
"""

from afmlkit.feature.core.structural_break.adf import (
    adf_test,
    adf_test_rolling,
)
from afmlkit.feature.core.structural_break.sadf import (
    sadf_test,
    SADFTest,
)
from afmlkit.feature.core.structural_break.cusum import (
    cusum_test_developing,
    cusum_test_last,
    cusum_test_rolling,
)

__all__ = [
    # ADF test
    'adf_test',
    'adf_test_rolling',
    # SADF test
    'sadf_test',
    'SADFTest',
    # CUSUM
    'cusum_test_developing',
    'cusum_test_last',
    'cusum_test_rolling',
]
