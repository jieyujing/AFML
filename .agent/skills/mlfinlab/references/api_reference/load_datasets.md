# API Reference: load_datasets.py

**Language**: Python

**Source**: `datasets/load_datasets.py`

---

## Functions

### load_stock_prices() → pd.DataFrame

Loads stock prices data sets consisting of
EEM, EWG, TIP, EWJ, EFA, IEF, EWQ, EWU, XLB, XLE, XLF, LQD, XLK, XLU, EPP, FXI, VGK, VPL, SPY, TLT, BND, CSJ,
DIA starting from 2008 till 2016.

:return: (pd.DataFrame) stock_prices data frame

**Returns**: `pd.DataFrame`



### load_tick_sample() → pd.DataFrame

Loads E-Mini S&P 500 futures tick data sample

:return: (pd.DataFrame) with tick data sample

**Returns**: `pd.DataFrame`



### load_dollar_bar_sample() → pd.DataFrame

Loads E-Mini S&P 500 futures dollar bars data sample.

:return: (pd.DataFrame) with dollar bar data sample

**Returns**: `pd.DataFrame`


