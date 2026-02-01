import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller

def analyze_normality(series):
    jb_stat, p_value = stats.jarque_bera(series)
    return jb_stat, p_value, series.skew(), series.kurtosis()

def analyze_stationarity(series):
    result = adfuller(series)
    return result[0], result[1]

df = pd.read_csv("data/output/dynamic_dollar_bars.csv", index_col=0)
df["close"] = np.log(df["close"])  # Log prices often used, but existing script used log returns on raw prices

# Re-calculate log returns from raw prices
# The CSV has raw OHL C
returns = np.log(df["close"] / df["close"].shift(1)).dropna()

jb, jb_p, skew, kurt = analyze_normality(returns)
adf, adf_p = analyze_stationarity(returns)

print(f"Stats for Dynamic Dollar Bars ({len(returns)} samples):")
print(f"JB Stat: {jb:.2f} (p={jb_p:.4f})")
print(f"Skew: {skew:.2f}")
print(f"Kurtosis: {kurt:.2f}")
print(f"ADF Stat: {adf:.2f}")
print(f"ADF p-value: {adf_p:.4f}")

if adf_p < 0.05:
    print("STATIONARY")
else:
    print("NON-STATIONARY")
