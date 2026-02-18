# API Reference: test_risk_estimators.py

**Language**: Python

**Source**: `tests/test_risk_estimators.py`

---

## Classes

### TestRiskEstimators

Tests different functions of the RiskEstimators class.

**Inherits from**: unittest.TestCase

#### Methods

##### setUp(self)

Initialize and get the test data

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_mp_pdf(self)

Test the deriving of pdf of the Marcenko-Pastur distribution.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_fit_kde(self)

Test the kernel fitting to a series of observations.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_pdf_fit(self)

Test the fit between empirical pdf and the theoretical pdf.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_find_max_eval(self)

Test the search for maximum random eigenvalue.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_corr_to_cov()

Test the recovering of the covariance matrix from the correlation matrix.

**Decorators**: `@staticmethod`


##### test_cov_to_corr()

Test the deriving of the correlation matrix from a covariance matrix.

**Decorators**: `@staticmethod`


##### test_get_pca()

Test the calculation of eigenvalues and eigenvectors from a Hermitian matrix.

**Decorators**: `@staticmethod`


##### test_denoised_corr()

Test the shrinkage the eigenvalues associated with noise.

**Decorators**: `@staticmethod`


##### test_denoised_corr_targ_shrink()

Test the second method of shrinkage of the eigenvalues associated with noise.

**Decorators**: `@staticmethod`


##### test_detoned()

Test the de-toning of the correlation matrix.

**Decorators**: `@staticmethod`


##### test_denoise_covariance()

Test the shrinkage the eigenvalues associated with noise.

**Decorators**: `@staticmethod`


##### test_minimum_covariance_determinant(self)

Test the calculation of the Minimum Covariance Determinant.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_empirical_covariance(self)

Test the calculation of the Maximum likelihood covariance estimator.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_shrinked_covariance(self)

Test the calculation of the Covariance estimator with shrinkage.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_semi_covariance(self)

Test the calculation of the Semi-Covariance matrix.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_exponential_covariance(self)

Test the calculation of the Exponentially-weighted Covariance matrix.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |



