# API Reference: backtests.py

**Language**: Python

**Source**: `backtest_statistics/backtests.py`

---

## Classes

### CampbellBacktesting

This class implements the Haircut Sharpe Ratios and Profit Hurdles algorithms described in the following paper:
`Campbell R. Harvey and Yan Liu, Backtesting, (Fall 2015). Journal of Portfolio Management,
2015 <https://papers.ssrn.com/abstract_id=2345489>`_; The code is based on the code provided by the authors of the paper.

The Haircut Sharpe Ratios algorithm lets the user adjust the observed Sharpe Ratios to take multiple testing into account
and calculate the corresponding haircuts. The haircut is the percentage difference between the original Sharpe ratio
and the new Sharpe ratio.

The Profit Hurdle algorithm lets the user calculate the required mean return for a strategy at a given level of
significance, taking multiple testing into account.

**Inherits from**: (none)

#### Methods

##### __init__(self, simulations = 2000)

Set the desired number of simulations to make in Haircut Sharpe Ratios or Profit Hurdle algorithms.

:param simulations: (int) Number of simulations

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| simulations | None | 2000 | - |


##### _sample_random_multest(rho, n_trails, prob_zero_mean, lambd, n_simulations, annual_vol = 0.15, n_obs = 240)

Generates empirical p-value distributions.

The algorithm is described in the paper and is based on the model estimated by `Harvey, C.R., Y. Liu,
and H. Zhu., … and the Cross-section of Expected Returns. Review of Financial Studies, forthcoming 2015`,
referred to as the HLZ model.

It provides a set of simulated t-statistics based on the parameters recieved from the _parameter_calculation
method.

Researchers propose a structural model to capture trading strategies’ underlying distribution.
With probability p0 (prob_zero_mean), a strategy has a mean return of zero and therefore comes
from the null distribution. With probability 1 – p0, a strategy has a nonzero mean and therefore
comes from the alternative distribution - exponential.

:param rho: (float) Average correlation among returns
:param n_trails: (int) Total number of trials inside a simulation
:param prob_zero_mean: (float) Probability for a random factor to have a zero mean
:param lambd: (float) Average of monthly mean returns for true strategies
:param n_simulations: (int) Number of rows (simulations)
:param annual_vol: (float) HLZ assume that the innovations in returns follow a normal distribution with a mean
                           of zero and a standard deviation of ma = 15%
:param n_obs: (int) Number of observations of used for volatility estimation from HLZ
:return: (np.ndarray) Array with distributions calculated

**Decorators**: `@staticmethod`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| rho | None | - | - |
| n_trails | None | - | - |
| prob_zero_mean | None | - | - |
| lambd | None | - | - |
| n_simulations | None | - | - |
| annual_vol | None | 0.15 | - |
| n_obs | None | 240 | - |


##### _parameter_calculation(rho)

Estimates the parameters used to generate the distributions in _sample_random_multest - the HLZ model.

Based on the work of HLZ, the pairwise correlation of returns is used to estimate the probability (prob_zero_mean),
total number of trials (n_simulations) and (lambd) - parameter of the exponential distribution. Levels and
parameters taken from the HLZ research.

:param rho: (float) Average correlation coefficient between strategy returns
:return: (np.array) Array of parameters

**Decorators**: `@staticmethod`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| rho | None | - | - |


##### _annualized_sharpe_ratio(sharpe_ratio, sampling_frequency = 'A', rho = 0, annualized = False, autocorr_adjusted = False)

Calculate the equivalent annualized Sharpe ratio after taking the autocorrelation of returns into account.

Adjustments are based on the work of `Lo, A., The Statistics of Sharpe Ratios. Financial Analysts Journal,
58 (2002), pp. 36-52` and are described there in more detail.

:param sharpe_ratio: (float) Sharpe ratio of the strategy
:param sampling_frequency: (str) Sampling frequency of returns
                           ['D','W','M','Q','A'] = [Daily, Weekly, Monthly, Quarterly, Annual]
:param rho: (float) Autocorrelation coefficient of returns at specified frequency
:param annualized: (bool) Flag if annualized, 'ind_an' = 1, otherwise = 0
:param autocorr_adjusted: (bool) Flag if Sharpe ratio was adjusted for returns autocorrelation
:return: (float) Adjusted annualized Sharpe ratio

**Decorators**: `@staticmethod`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| sharpe_ratio | None | - | - |
| sampling_frequency | None | 'A' | - |
| rho | None | 0 | - |
| annualized | None | False | - |
| autocorr_adjusted | None | False | - |


##### _monthly_observations(num_obs, sampling_frequency)

Calculates the number of monthly observations based on sampling frequency and number of observations.

:param num_obs: (int) Number of observations used for modelling
:param sampling_frequency: (str) Sampling frequency of returns
                           ['D','W','M','Q','A'] = [Daily, Weekly, Monthly, Quarterly, Annual]
:return: (np.float64) Number of monthly observations

**Decorators**: `@staticmethod`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| num_obs | None | - | - |
| sampling_frequency | None | - | - |


##### _holm_method_sharpe(all_p_values, num_mult_test, p_val)

Runs one cycle of the Holm method for the Haircut Shape ratio algorithm.

:param all_p_values: (np.array) Sorted p-values to adjust
:param num_mult_test: (int) Number of multiple tests allowed
:param p_val: (float) Significance level p-value
:return: (np.float64) P-value adjusted at a significant level

**Decorators**: `@staticmethod`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| all_p_values | None | - | - |
| num_mult_test | None | - | - |
| p_val | None | - | - |


##### _bhy_method_sharpe(all_p_values, num_mult_test, p_val)

Runs one cycle of the BHY method for the Haircut Shape ratio algorithm.

:param all_p_values: (np.array) Sorted p-values to adjust
:param num_mult_test: (int) Number of multiple tests allowed
:param p_val: (float) Significance level p-value
:param c_constant: (float) Constant used in BHY method
:return: (np.float64) P-value adjusted at a significant level

**Decorators**: `@staticmethod`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| all_p_values | None | - | - |
| num_mult_test | None | - | - |
| p_val | None | - | - |


##### _sharpe_ratio_haircut(p_val, monthly_obs, sr_annual)

Calculates the adjusted Sharpe ratio and the haircut based on the final p-value of the method.

:param p_val: (float) Adjusted p-value of the method
:param monthly_obs: (int) Number of monthly observations
:param sr_annual: (float) Annualized Sharpe ratio to compare to
:return: (np.array) Elements (Adjusted annual Sharpe ratio, Haircut percentage)

**Decorators**: `@staticmethod`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| p_val | None | - | - |
| monthly_obs | None | - | - |
| sr_annual | None | - | - |


##### _holm_method_returns(p_values_simulation, num_mult_test, alpha_sig)

Runs one cycle of the Holm method for the Profit Hurdle algorithm.

:param p_values_simulation: (np.array) Sorted p-values to adjust
:param num_mult_test: (int) Number of multiple tests allowed
:param alpha_sig: (float) Significance level (e.g., 5%)
:return: (np.float64) P-value adjusted at a significant level

**Decorators**: `@staticmethod`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| p_values_simulation | None | - | - |
| num_mult_test | None | - | - |
| alpha_sig | None | - | - |


##### _bhy_method_returns(p_values_simulation, num_mult_test, alpha_sig)

Runs one cycle of the BHY method for the Profit Hurdle algorithm.

:param p_values_simulation: (np.array) Sorted p-values to adjust
:param num_mult_test: (int) Number of multiple tests allowed
:param alpha_sig: (float) Significance level (e.g., 5%)
:return: (np.float64) P-value adjusted at a significant level

**Decorators**: `@staticmethod`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| p_values_simulation | None | - | - |
| num_mult_test | None | - | - |
| alpha_sig | None | - | - |


##### haircut_sharpe_ratios(self, sampling_frequency, num_obs, sharpe_ratio, annualized, autocorr_adjusted, rho_a, num_mult_test, rho)

Calculates the adjusted Sharpe ratio due to testing multiplicity.

This algorithm lets the user calculate Sharpe ratio adjustments and the corresponding haircuts based on
the key parameters of returns from the strategy. The adjustment methods are Bonferroni, Holm,
BHY (Benjamini, Hochberg and Yekutieli) and the Average of them. The algorithm calculates adjusted p-value,
adjusted Sharpe ratio and the haircut.

The haircut is the percentage difference between the original Sharpe ratio and the new Sharpe ratio.

:param sampling_frequency: (str) Sampling frequency ['D','W','M','Q','A'] of returns
:param num_obs: (int) Number of returns in the frequency specified in the previous step
:param sharpe_ratio: (float) Sharpe ratio of the strategy. Either annualized or in the frequency specified in the previous step
:param annualized: (bool) Flag if Sharpe ratio is annualized
:param autocorr_adjusted: (bool) Flag if Sharpe ratio was adjusted for returns autocorrelation
:param rho_a: (float) Autocorrelation coefficient of returns at the specified frequency (if the Sharpe ratio
                      wasn't corrected)
:param num_mult_test: (int) Number of other strategies tested (multiple tests)
:param rho: (float) Average correlation among returns of strategies tested
:return: (np.ndarray) Array with adjuted p-value, adjusted Sharpe ratio, and haircut as rows
                      for Bonferroni, Holm, BHY and average adjustment as columns

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| sampling_frequency | None | - | - |
| num_obs | None | - | - |
| sharpe_ratio | None | - | - |
| annualized | None | - | - |
| autocorr_adjusted | None | - | - |
| rho_a | None | - | - |
| num_mult_test | None | - | - |
| rho | None | - | - |


##### profit_hurdle(self, num_mult_test, num_obs, alpha_sig, vol_anu, rho)

Calculates the required mean monthly return for a strategy at a given level of significance.

This algorithm uses four adjustment methods - Bonferroni, Holm, BHY (Benjamini, Hochberg and Yekutieli)
and the Average of them. The result is the Minimum Average Monthly Return for the strategy to be significant
at a given significance level, taking into account multiple testing.

This function doesn't allow for any autocorrelation in the strategy returns.

:param num_mult_test: (int) Number of tests in multiple testing allowed (number of other strategies tested)
:param num_obs: (int) Number of monthly observations for a strategy
:param alpha_sig: (float) Significance level (e.g., 5%)
:param vol_anu: (float) Annual volatility of returns(e.g., 0.05 or 5%)
:param rho: (float) Average correlation among returns of strategies tested
:return: (np.ndarray) Minimum Average Monthly Returns for
                      [Independent tests, Bonferroni, Holm, BHY and Average for Multiple tests]

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| num_mult_test | None | - | - |
| num_obs | None | - | - |
| alpha_sig | None | - | - |
| vol_anu | None | - | - |
| rho | None | - | - |



