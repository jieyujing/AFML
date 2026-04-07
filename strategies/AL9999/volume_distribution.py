"""
# @Author ：Benoni
# @version：Foucs_1.0
"""
import pandas as pd
import numpy as np
from functools import partial


class BaseStrategy:
    @staticmethod
    def calculate(df, *args, **kwargs):
        raise NotImplementedError()

class QIML0116:
    """
    对局部峰值进行度量，局部峰值可以反应对趋势交易者或者知情交易者的交易行为的刻画
    理论上是个正向因子

    tips: 这里用的是总量，但是币上是有buy sell volume的，可以分开计算，最后整合 (a - b） / (a + b)
    """
    @staticmethod
    def vol_diff_std(x):
        xx = x['amount'] / x['count']
        return (xx.diff() / xx.mean()).abs().std()

    @staticmethod
    def calculate(df, frequency='H', windows=20):
        factors = df.groupby(['code']).resample(frequency).apply(QIML0116.vol_diff_std)
        factors_mean = factors.groupby(['code']).rolling(windows).mean().droplevel(0)
        return factors, factors_mean


class QIML0124:
    """
    刻画放量，也是反应趋势交易和知情交易者

    tips: 这里用的是总量，但是币上是有buy sell volume的，可以分开计算，最后整合 (a - b） / (a + b)
    """
    @staticmethod
    def vol_peak(x):
        windows = len(x['amount'])
        x_mean, x_std = x['amount'].mean(), x['amount'].std()
        x_sub = x['amount'][x['amount'] > (x_mean + x_std)].reset_index()
        count = len(x_sub.diff().query('index>1')) + 1
        return count / windows

    @staticmethod
    def calculate(df, frequency='H', windows=20):
        factors = df.groupby(['code']).resample(frequency).apply(QIML0124.vol_peak)
        factors_mean = factors.groupby(['code']).rolling(windows).mean().droplevel(0)
        return factors, factors_mean

class QIML0203:
    """
    这个鸟因子，考虑投资者的注意，用股票代码排序。。。。比如 000001  000002
    不能说完全没用吧，但只能说有点玄学，故 pass
    """
    pass

class QIML0212:
    """
    衡量投资者因厌恶波动模糊而急于平仓时所付出的流动性成本
    这里涉及到截面计算，因此不同的品种池，可能最后因子值不一样
    """

    @staticmethod
    def ambiguity(x):
        x_ret = x['close'].pct_change()
        x_amb = x_ret.rolling(5).std().rolling(5).std()
        x_fogging = x[x_amb > x_amb.mean()]
        x_amb_ratio = x_fogging['volume'].mean() / x['volume'].mean()
        x_amb_ratio_1 = x_fogging['amount'].mean() / x['amount'].mean()
        x_amb_ratio_last = x_amb_ratio - x_amb_ratio_1
        return x_amb_ratio_last

    @staticmethod
    def calculate(df, frequency='H'):
        factors = df.groupby(['code']).resample(frequency).apply(QIML0212.ambiguity)
        s1 = factors.groupby('time').apply(lambda x: x[x < 0].sum()).rename('s1').reset_index()
        factors[factors < 0] /= factors.groupby('code').rolling(10).std().droplevel(0).fillna(1.)
        s2 = factors.groupby('time').apply(lambda x: x[x < 0].sum()).rename('s2').reset_index()
        factors = factors.reset_index().merge(s1).merge(s2)
        factors.columns = ['code', 'time', 'factors', 's1', 's2']
        factors['factors'] = np.where(factors['factors'] < 0, factors['factors'] / factors['s2'] * factors['s1'], factors['factors'])
        factors = factors[['code', 'time', 'factors']]
        return factors

class QIML0301:
    """
    统计投资者在模糊较大的时的成交程度，同时衡量了对模糊性的厌恶程度

    tips: 这里用的是总量，但是币上是有buy sell volume的，可以分开计算，最后整合 (a - b） / (a + b)
    """
    @staticmethod
    def ambiguity(x):
        x_ret = x['close'].pct_change()
        x_amb = x_ret.rolling(5).std().rolling(5).std()
        x_fogging = x[x_amb > x_amb.mean()]
        x_amb_ratio = x_fogging['volume'].mean() / x['volume'].mean()
        return x_amb_ratio

    @staticmethod
    def calculate(df, frequency='H'):
        factors = df.groupby(['code']).resample(frequency).apply(QIML0301.ambiguity)
        return factors

class QIML0331:
    """
    统计投资者在模糊较大的时的成交程度，同时衡量了对模糊性的厌恶程度

    tips: 这里用的是总量，但是币上是有buy sell count的，可以分开计算，最后整合 (a - b） / (a + b)
          也可以用amount   但其实差不多  大部分币上的count 和amount corr都 0.97+
    """
    @staticmethod
    def ambiguity(x):
        x_ret = x['close'].pct_change()
        x_amb = x_ret.rolling(5).std().rolling(5).std()
        x_fogging = x[x_amb > x_amb.mean()]
        x_amb_ratio = x_fogging['count'].mean() / x['count'].mean()
        return x_amb_ratio

    @staticmethod
    def calculate(df, frequency='H'):
        factors = df.groupby(['code']).resample(frequency).apply(QIML0331.ambiguity)
        return factors

class QIML0401:
    """
    volume profile 的 p shape

    """

    @staticmethod
    def p_shape(x):
        vol_sum = x.groupby('close')['amount'].sum()
        vol_acc_sum = vol_sum.sum()
        idx = np.argmax(vol_sum)
        ratio = vol_sum.iloc[idx] / vol_acc_sum
        num = 0
        while ratio < 0.5:
            num += 1
            ratio = vol_acc_sum.iloc[idx - num: idx + num].sum() / vol_acc_sum
        try:
            vsa_low = vol_sum.index[idx - num]
        except:
            vsa_low = np.min(vol_sum.index)
        vsa_low2max = (x['close'].max() - vsa_low) / vsa_low
        return vsa_low2max

    @staticmethod
    def calculate(df, frequency='H'):
        factors = df.groupby(['code']).resample(frequency).apply(QIML0401.p_shape)
        return factors

class QIML0413:
    """
    volume profile 的 b shape

    """
    @staticmethod
    def b_shape(x):
        vol_sum = x.groupby('close')['amount'].sum()
        vol_acc_sum = vol_sum.sum()
        idx = np.argmax(vol_sum)
        ratio = vol_sum.iloc[idx] / vol_acc_sum
        num = 0
        while ratio < 0.5:
            num += 1
            ratio = vol_acc_sum.iloc[idx - num: idx + num].sum() / vol_acc_sum
        try:
            vsa_high = vol_sum.index[idx + num]
        except:
            vsa_high = np.max(vol_sum.index)
        vsa_high2min = (vsa_high - x['close'].min()) / vsa_high
        return vsa_high2min

    @staticmethod
    def calculate(df, frequency='H'):
        factors = df.groupby(['code']).resample(frequency).apply(QIML0413.b_shape)
        return factors

class QIML0503:
    """
    volume profile 中  对比close 的差异
    不知道为什么因子日历中没有对比high 与 close的差异
    """
    @staticmethod
    def p_shape(x):
        vol_sum = x.groupby('close')['amount'].sum()
        vol_acc_sum = vol_sum.sum()
        idx = np.argmax(vol_sum)
        ratio = vol_sum.iloc[idx] / vol_acc_sum
        num = 0
        while ratio < 0.5:
            num += 1
            ratio = vol_acc_sum.iloc[idx - num: idx + num].sum() / vol_acc_sum
        try:
            vsa_low = vol_sum.index[idx - num]
        except:
            vsa_low = np.min(vol_sum.index)
        vsa_low2close = (x['close'] - vsa_low) / vsa_low
        return vsa_low2close

    @staticmethod
    def calculate(df, frequency='H'):
        factors = df.groupby(['code']).resample(frequency).apply(QIML0503.p_shape)
        return factors

class QIML0514:
    """
    成交量不稳定性的分散程度
    """
    @staticmethod
    def vol_entropy(x):
        bins = min(5, len(x['amount'].unique()))
        b = pd.value_counts(pd.cut(x['amount'], bins))
        b = b[b > 0] / len(x['amount'])
        vol_ent = - (b * np.log(b)).sum()
        return vol_ent

    @staticmethod
    def calculate(df, frequency='H'):
        factors = df.groupby(['code']).resample(frequency).apply(QIML0514.vol_entropy)
        factors_std = factors.groupby('code').rolling(24).std().droplevel(0)
        return factors, factors_std

class QIML0607:
    """
    成交量占比偏度
    tips: 对数成交量 换手率 收益率 都可以当做变量
    """

    @staticmethod
    def vol_skew(x):
        kurt = (x.resample('5min')['amount'].sum().dropna() / x['amount'].sum()).skew()
        return kurt

    @staticmethod
    def calculate(df, frequency='H'):
        factors = df.groupby(['code']).resample(frequency).apply(QIML0607.vol_skew)
        factors_mean = factors.groupby('code').rolling(24).mean().droplevel(0)
        return factors, factors_mean

class QIML0618:
    """
    成交量占比峰度
    tips: 对数成交量 换手率 收益率 都可以当做变量
    """

    @staticmethod
    def vol_kurt(x):
        kurt = (x.resample('5min')['amount'].sum().dropna() / x['amount'].sum()).kurt()
        return kurt

    @staticmethod
    def calculate(df, frequency='H'):
        factors = df.groupby(['code']).resample(frequency).apply(QIML0618.vol_kurt)
        factors_mean = factors.groupby('code').rolling(24).mean().droplevel(0)
        return factors, factors_mean

class QIML0629:
    """
    一致买入交易 一致行为，趋势显著
    这里对比原版做了点修改，原版是只看多头 但合约多空都可以做  所以加了空头
    tips： 还可以把这里的amount换掉，换成对应的多空 amount
    """
    @staticmethod
    def unanimous_buying(x):
        alpha = abs(x['close'] - x['open']) / abs(x['high'] - x['low'])
        alpha = alpha.fillna(0)
        vol_up = x['amount'][(alpha > 0.5) & (x['close'].pct_change() > 0)].sum() / x['amount'].sum()
        vol_down = x['amount'][(alpha > 0.5) & (x['close'].pct_change() < 0)].sum() / x['amount'].sum()
        return (vol_up + vol_down) / (vol_up - vol_down)

    @staticmethod
    def calculate(df, frequency='H'):
        factors = df.groupby(['code']).resample(frequency).apply(QIML0629.unanimous_buying)
        factors_mean = factors.groupby('code').rolling(24).mean().droplevel(0)
        return factors, factors_mean

class QIML0708:
    """
    这个因子需要用到tick数据，目前暂时不考虑逐笔成交
    故 pass
    tips：
    """
    pass

class QIML0722:
    """
      成交量占比标准差
      tips: 对数成交量 换手率 收益率 都可以当做变量
    """

    @staticmethod
    def vol_std(x):
        std = (x.resample('5min')['amount'].sum().dropna() / x['amount'].sum()).std()
        return std

    @staticmethod
    def calculate(df, frequency='H'):
        factors = df.groupby(['code']).resample(frequency).apply(QIML0722.vol_std)
        factors_mean = factors.groupby('code').rolling(24).mean().droplevel(0)
        return factors, factors_mean

class QIML0806:
    """
      分钟成交额方差
      tips: 对数成交量 换手率 收益率 都可以当做变量
    """
    @staticmethod
    def calculate(df, frequency='H'):
        factors = df.groupby(['code']).resample(frequency).agg({'volume': 'var'})
        return factors

class QIML0820:
    """
    这个因子需要用到tick数据，目前暂时不考虑逐笔成交
    故 pass
    tips：
    """
    pass

class QIML0827:
    """
    一致交易 一致行为，趋势显著
    对比一致买入，就是多空算到一起了
    """

    @staticmethod
    def unanimous_trading(x):
        alpha = abs(x['close'] - x['open']) / abs(x['high'] - x['low'])
        alpha = alpha.fillna(0)
        vol_ = x['amount'][(alpha > 0.5)].sum() / x['amount'].sum()
        return vol_

    @staticmethod
    def calculate(df, frequency='H'):
        factors = df.groupby(['code']).resample(frequency).apply(QIML0827.unanimous_trading)
        factors_mean = factors.groupby('code').rolling(24).mean().droplevel(0)
        return factors, factors_mean

class QIML0903:
    """
    持续异常成交量
    """

    @staticmethod
    def atv(df, frequency):
        pre_volume = df.groupby(['code']).resample(frequency).agg({'amount': 'mean'}).shift()
        pre_volume = pre_volume.reset_index()
        return pre_volume

    @staticmethod
    def ATV(x, pre_volume):
        pre_volum_values = pre_volume[(pre_volume['code'] == x['code'][0]) & (pre_volume['time'] == x['time'][0])]['amount']
        if len(pre_volum_values) > 0:
            return x['amount'] / pre_volum_values.values

    @staticmethod
    def cal_daily_patv(x):
        PATV = x[x.columns[-1]].mean() / x[x.columns[-1]].std() + x[x.columns[-1]].kurt()
        return PATV

    @staticmethod
    def calculate(df, frequency='H'):
        pre_volume = QIML0903.atv(df, frequency)
        atv_func = partial(QIML0903.ATV, pre_volume=pre_volume)
        factors = df.groupby('code').resample(frequency).apply(atv_func)
        factors = factors.dropna().groupby('time').apply(lambda x: x.rank(pct=True)).droplevel(level=2).reset_index()
        factors.index = pd.to_datetime(factors['time'])
        factors = factors.groupby('code').resample(frequency).apply(QIML0903.cal_daily_patv)
        factors_mean = factors.groupby('code').rolling(24).mean().droplevel(0)
        return factors, factors_mean

class QIML0914:
    """
    尾盘成交额占比
    这里只计算最后5分钟
    """

    @staticmethod
    def apl(x):
        x['time'] = pd.to_datetime(x['time'])
        cp = x[x.index.dt.minute >= 55]['volume'].sum() / (x['volume'].sum())
        cp = cp.fillna(0)
        return cp

    @staticmethod
    def calculate(df, frequency='H'):
        factors = df.groupby(['code']).resample(frequency).apply(QIML0914.apl)
        factors_mean = factors.groupby('code').rolling(24).mean().droplevel(0)
        return factors, factors_mean

class QIML1003:
    """
    异常成交量
    """

    @staticmethod
    def atv(df, frequency):
        pre_volume = df.groupby(['code']).resample(frequency).agg({'amount': 'mean'}).shift()
        pre_volume = pre_volume.reset_index(inplace=True)
        return pre_volume

    @staticmethod
    def ATV(x, pre_volume):
        pre_volum_values = pre_volume[(pre_volume['code'] == x['code'][0]) & (pre_volume['time'] == x['time'][0])][
            'amount']
        if len(pre_volum_values) > 0:
            return (x['amount'] / pre_volum_values.values).mean()

    @staticmethod
    def calculate(df, frequency='H'):
        pre_volume = QIML1003.atv(df, frequency)
        atv_func = partial(QIML1003.ATV, pre_volume=pre_volume)
        factors = df.groupby('code').resample(frequency).apply(atv_func)
        return factors

class QIML1014:
    """
    成交量比值
    计算开盘前5分钟和后5分钟的比值
    """

    @staticmethod
    def vr(x):
        x['time'] = pd.to_datetime(x['time'])
        op = x[x['time'].dt.minute < 5]['amount'].sum() / (x['amount'].sum())
        cp = x[x['time'].dt.minute >= 55]['amount'].sum() / (x['amount'].sum())
        op = op.fillna(0)
        cp = cp.fillna(0)
        return op / cp

    @staticmethod
    def calculate(df, frequency='H'):
        factors = df.groupby(['code']).resample(frequency).apply(QIML1014.vr)
        factors_mean = factors.groupby('code').rolling(24).mean().droplevel(0)
        return factors, factors_mean

class QIML1021:
    """
    成交量占比
    计算开盘前5分钟和后5分钟的占比
    """

    @staticmethod
    def op_cp(x):
        x['time'] = pd.to_datetime(x['time'])
        op = x[x['time'].dt.minute < 5]['amount'].sum() / (x['amount'].sum())
        cp = x[x['time'].dt.minute >= 55]['amount'].sum() / (x['amount'].sum())
        op = op.fillna(0)
        cp = cp.fillna(0)
        return op + cp

    @staticmethod
    def calculate(df, frequency='H'):
        factors = df.groupby(['code']).resample(frequency).apply(QIML1021.op_cp)
        factors_mean = factors.groupby('code').rolling(24).mean().droplevel(0)
        return factors, factors_mean

class QIML1105:
    """
    单笔成交金额分位数
    """

    @staticmethod
    def qua(x):
        x_sort = (x['volume'] / x['amount']).sort_values()
        x_sort = x_sort[:-2]
        qua = (x_sort.quantile(0.1) - x_sort.min()) / (x_sort.max() - x_sort.min())
        qua = qua.fillna(0)
        return qua

    @staticmethod
    def calculate(df, frequency='H'):
        factors = df.groupby(['code']).resample(frequency).apply(QIML1105.qua)
        factors_mean = factors.groupby('code').rolling(24).mean().droplevel(0)
        return factors, factors_mean

class QIML1113:
    """
    主力交易强度
    """

    @staticmethod
    def ts(x):
        return (x['volume'] / x['amount']).corr(x['volume'], method='spearman')

    @staticmethod
    def calculate(df, frequency='H'):
        factors = df.groupby(['code']).resample(frequency).apply(QIML1113.ts)
        factors_mean = factors.groupby('code').rolling(24).mean().droplevel(0)
        return factors, factors_mean

class QIML1202:
    """
    一致卖出交易
    我这里和一致买入写到一起了，故pass
    """
    pass

class QIML1215:
    """
    尾盘成交占比
    和成交额占比没太大区别，只是从money换成了量
    """

    @staticmethod
    def apl(x):
        x['time'] = pd.to_datetime(x['time'])
        cp = x[x.index.dt.minute >= 55]['amount'].sum() / (x['amount'].sum())
        cp = cp.fillna(0)
        return cp

    @staticmethod
    def calculate(df, frequency='H'):
        factors = df.groupby(['code']).resample(frequency).apply(QIML1215.apl)
        factors_mean = factors.groupby('code').rolling(24).mean().droplevel(0)
        return factors, factors_mean


class QIML1222:
    """
    换手率分布均匀度
    """

    @staticmethod
    def tvd(x):
        return (x['amount'] / x['amount'].sum()).std().fillna(0)

    @staticmethod
    def calculate(df, frequency='H'):
        factors = df.groupby(['code']).resample(frequency).apply(QIML1222.tvd)
        factors_mean = factors.groupby('code').rolling(24).mean().droplevel(0)
        factors_std = factors.groupby('code').rolling(24).std().droplevel(0)
        return factors, factors_std / factors_mean


def get_factory(factor_name):
    factors = {
        'QIML0116': QIML0116,
        'QIML0124': QIML0124,
        'QIML0212': QIML0212,
        'QIML0301': QIML0301,
        'QIML0331': QIML0331,
        'QIML0401': QIML0401,
        'QIML0413': QIML0413,
        'QIML0503': QIML0503,
        'QIML0514': QIML0514,
        'QIML0607': QIML0607,
        'QIML0618': QIML0618,
        'QIML0629': QIML0629,
        'QIML0722': QIML0722,
        'QIML0806': QIML0806,
        'QIML0827': QIML0827,
        'QIML0903': QIML0903,
        'QIML0914': QIML0914,
        'QIML1003': QIML1003,
        'QIML1014': QIML1014,
        'QIML1021': QIML1021,
        'QIML1105': QIML1105,
        'QIML1113': QIML1113,
        'QIML1215': QIML1215,
        'QIML1222': QIML1222,

    }
    return factors.get(factor_name, BaseStrategy)


def volume_distribution(data, strategy_name='QIML0116', freq='H', window=24):

    strategy_class = get_factory(strategy_name)
    factors, factors_mean = strategy_class.calculate(data, frequency=freq, windows=window)

    if factors is None:
        factors = pd.DataFrame()
    if factors_mean is None:
        factors_mean = pd.DataFrame()

    return factors, factors_mean
