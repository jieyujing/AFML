# API Reference: codependence_matrix.py

**Language**: Python

**Source**: `codependence/codependence_matrix.py`

---

## Functions

### get_dependence_matrix(df: pd.DataFrame, dependence_method: str, theta: float = 0.5, bandwidth: float = 0.01) → pd.DataFrame

This function returns a dependence matrix for elements given in the dataframe using the chosen dependence method.

List of supported algorithms to use for generating the dependence matrix: ``information_variation``,
``mutual_information``, ``distance_correlation``, ``spearmans_rho``, ``gpr_distance``, ``gnpr_distance``.

:param df: (pd.DataFrame) Features.
:param dependence_method: (str) Algorithm to be use for generating dependence_matrix.
:param theta: (float) Type of information being tested in the GPR and GNPR distances. Falls in range [0, 1].
                      (0.5 by default)
:param bandwidth: (float) Bandwidth to use for splitting observations in the GPR and GNPR distances. (0.01 by default)
:return: (pd.DataFrame) Dependence matrix.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| df | pd.DataFrame | - | - |
| dependence_method | str | - | - |
| theta | float | 0.5 | - |
| bandwidth | float | 0.01 | - |

**Returns**: `pd.DataFrame`



### get_distance_matrix(X: pd.DataFrame, distance_metric: str = 'angular') → pd.DataFrame

Applies distance operator to a dependence matrix.

This allows to turn a correlation matrix into a distance matrix. Distances used are true metrics.

List of supported distance metrics to use for generating the distance matrix: ``angular``, ``squared_angular``,
and ``absolute_angular``.

:param X: (pd.DataFrame) Dataframe to which distance operator to be applied.
:param distance_metric: (str) The distance metric to be used for generating the distance matrix.
:return: (pd.DataFrame) Distance matrix.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| X | pd.DataFrame | - | - |
| distance_metric | str | 'angular' | - |

**Returns**: `pd.DataFrame`


