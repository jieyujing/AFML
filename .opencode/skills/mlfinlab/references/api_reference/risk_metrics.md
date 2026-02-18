# API Reference: risk_metrics.py

**Language**: Python

**Source**: `portfolio_optimization/risk_metrics.py`

---

## Classes

### RiskMetrics

This class contains methods for calculating common risk metrics used in trading and asset management.

**Inherits from**: (none)

#### Methods

##### __init__(self)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### calculate_variance(covariance, weights)

Calculate the variance of a portfolio.

:param covariance: (pd.DataFrame/np.matrix) Covariance matrix of assets
:param weights: (list) List of asset weights
:return: (float) Variance of a portfolio

**Decorators**: `@staticmethod`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| covariance | None | - | - |
| weights | None | - | - |


##### calculate_value_at_risk(returns, confidence_level = 0.05)

Calculate the value at risk (VaR) of a portfolio/asset.

:param returns: (pd.DataFrame/np.array) Historical returns for an asset / portfolio
:param confidence_level: (float) Confidence level (alpha)
:return: (float) VaR

**Decorators**: `@staticmethod`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| returns | None | - | - |
| confidence_level | None | 0.05 | - |


##### calculate_expected_shortfall(self, returns, confidence_level = 0.05)

Calculate the expected shortfall (CVaR) of a portfolio/asset.

:param returns: (pd.DataFrame/np.array) Historical returns for an asset / portfolio
:param confidence_level: (float) Confidence level (alpha)
:return: (float) Expected shortfall

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| returns | None | - | - |
| confidence_level | None | 0.05 | - |


##### calculate_conditional_drawdown_risk(returns, confidence_level = 0.05)

Calculate the conditional drawdown of risk (CDaR) of a portfolio/asset.

:param returns: (pd.DataFrame/np.array) Historical returns for an asset / portfolio
:param confidence_level: (float) Confidence level (alpha)
:return: (float) Conditional drawdown risk

**Decorators**: `@staticmethod`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| returns | None | - | - |
| confidence_level | None | 0.05 | - |



