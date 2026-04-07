"""
# @Author ：Benoni
# @version：Foucs_1.0
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class BaseStrategy:
    @staticmethod
    def calculate(df, *args, **kwargs):
        raise NotImplementedError()


class QIML0109:
    """
    主动买卖因子
    原版研报：超大单（>100万元）对应机构，大单（20-100万）对应大户，中单（4-20万）对应中户，小单（<4万）对应散户
    这里感觉很拍脑袋这个想法，既然这样，这里直接修改，用Kmeans聚类算法
    这个写法，运行速度上有点慢 (暂时没有加入 日历中的2和3的计算， 因为发现这边计算的corr还行)
    感觉加入了应该会更好，到时候改吧

    算的比较慢，这个是个问题，到时候一起解决吧
    毕竟用了Kmeans

    记得要修改！！！！！！！！！！！！！！！！！！！
    """

    @staticmethod
    def act(x):
        x_copy = x.copy()
        x_copy['sell_volume'] = x_copy['volume'] - x['buy_volume']
        for volume_type in ['buy_volume', 'sell_volume']:
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x_copy[volume_type].values.reshape(-1, 1))
            kmeans = KMeans(n_clusters=3, random_state=42)
            x_copy[f'{volume_type}_cluster'] = kmeans.fit_predict(x_scaled)

            volume_list = []
            for cluster_num in range(3):
                cluster_data = x_copy[x_copy[f'{volume_type}_cluster'] == cluster_num]['volume'].mean()
                volume_list.append(cluster_data)
                x_copy.loc[x_copy[f'{volume_type}_cluster'] == cluster_num, f'{volume_type}_cluster'] = cluster_data

            x_copy.loc[x_copy[f'{volume_type}_cluster'] == max(volume_list), f'{volume_type}_cluster'] = f'{volume_type}_big'
            x_copy.loc[x_copy[f'{volume_type}_cluster'] == min(volume_list), f'{volume_type}_cluster'] = f'{volume_type}_small'
            x_copy.loc[x_copy[f'{volume_type}_cluster'] == np.median(volume_list), f'{volume_type}_cluster'] = f'{volume_type}_median'

        buy_big_median = x_copy[x_copy['buy_volume_cluster'] == 'buy_volume_big']['buy_volume'].sum() + x_copy[x_copy['buy_volume_cluster'] == 'buy_volume_median']['buy_volume'].sum()
        buy_sell_median = x_copy[x_copy['sell_volume_cluster'] == 'sell_volume_big']['sell_volume'].sum() + x_copy[x_copy['sell_volume_cluster'] == 'sell_volume_median']['sell_volume'].sum()
        buy_small = x_copy[x_copy['buy_volume_cluster'] == 'buy_volume_small']['buy_volume'].sum()
        sell_small = x_copy[x_copy['sell_volume_cluster'] == 'sell_volume_small']['sell_volume'].sum()

        act_up = (buy_big_median - buy_sell_median) / (buy_big_median + buy_sell_median)
        act_down = (buy_small - sell_small) / (buy_small + sell_small)
        return act_up, act_down

    @staticmethod
    def calculate(df, frequency='H', windows=24):
        factors = df.groupby(['code']).resample(frequency).apply(QIML0109.act)
        factors = pd.DataFrame(factors.tolist(), index=factors.index)
        factors.columns = ['act_up', 'act_down']
        factors_mean = factors.groupby(['code']).rolling(windows).mean().droplevel(0)
        return factors, factors_mean


class QIML0114:
    """
    动态知情交易概率
    这个好复杂, 先pass吧
    """
    pass


class QIML0120:
    """
    残差资金流强度因子
    看到残差，没错，回归, 先pass，后面补
    """
    pass


class QIML0130:
    """
    价格冲击偏差
    朱剑涛的也挺难写的，先pass吧
    """
    pass


class QIML0208:
    """
    超大单和小单的同步相关性
    也是算的有点慢，rolling.corr() 不能给method
    只能自己for循环了
    """

    @staticmethod
    def rankcorr(x):
        x_copy = x.copy()
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_copy['volume'].values.reshape(-1, 1))
        kmeans = KMeans(n_clusters=3, random_state=42)
        x_copy['volume_cluster'] = kmeans.fit_predict(x_scaled)

        volume_list = []
        for cluster_num in range(3):
            cluster_data = x_copy[x_copy['volume_cluster'] == cluster_num]['volume'].mean()
            volume_list.append(cluster_data)
            x_copy.loc[x_copy['volume_cluster'] == cluster_num, 'volume_cluster'] = cluster_data

        x_copy.loc[x_copy['volume_cluster'] == max(volume_list), 'volume_cluster'] = 'volume_cluster_big'
        x_copy.loc[x_copy['volume_cluster'] == min(volume_list), 'volume_cluster'] = 'volume_cluster_small'

        big_volume = x_copy[x_copy['volume_cluster'] == 'volume_cluster_big']['volume'].sum()
        small_volume = x_copy[x_copy['volume_cluster'] == 'volume_cluster_small']['volume'].sum()

        return big_volume, small_volume

    @staticmethod
    def calculate(df, frequency='H', windows=24):
        factors = df.groupby(['code']).resample(frequency).apply(QIML0208.rankcorr)
        factors = pd.DataFrame(factors.tolist(), index=factors.index)
        factors.columns = ['big_volume', 'small_volume']

        def rolling_spearmanr(x):
            result = np.empty(len(x['big_volume']), dtype=float)
            result.fill(np.nan)

            for i in range(windows - 1, len(x['big_volume'])):
                x_window = x['big_volume'][i - windows + 1: i + 1]
                y_window = x['small_volume'][i - windows + 1: i + 1]
                result[i] = np.corrcoef(np.argsort(x_window), np.argsort(y_window), rowvar=False)[0, 1]
            return pd.Series(result, index=x.index)

        factors = factors.groupby(['code']).apply(rolling_spearmanr).droplevel(0)
        factors_mean = factors.groupby(['code']).rolling(windows).mean().droplevel(0)
        return factors, factors_mean


class QIML0217:
    """
    小单和小单的错位相关性
    实在写不动了   明天再来吧
    """
    @staticmethod
    def rankcorr(x):
        x_copy = x.copy()
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_copy['volume'].values.reshape(-1, 1))
        kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
        x_copy['volume_cluster'] = kmeans.fit_predict(x_scaled)

        volume_list = []
        for cluster_num in range(3):
            cluster_data = x_copy[x_copy['volume_cluster'] == cluster_num]['volume'].mean()
            volume_list.append(cluster_data)
            x_copy.loc[x_copy['volume_cluster'] == cluster_num, 'volume_cluster'] = cluster_data

        x_copy.loc[x_copy['volume_cluster'] == max(volume_list), 'volume_cluster'] = 'volume_cluster_big'
        x_copy.loc[x_copy['volume_cluster'] == min(volume_list), 'volume_cluster'] = 'volume_cluster_small'

        small_volume = x_copy[x_copy['volume_cluster'] == 'volume_cluster_small']['volume'].sum()

        return small_volume

    @staticmethod
    def calculate(df, frequency='D', windows=24):
        factors = df.groupby(['code']).resample(frequency).apply(QIML0217.rankcorr)
        def rolling_spearmanr(x):
            result = np.empty(len(x), dtype=float)
            result.fill(np.nan)

            for i in range(windows - 1, len(x)-1):
                x_window = x[i - windows + 1: i + 1]
                y_window = x[i - windows + 2: i + 2]
                result[i] = np.corrcoef(np.argsort(x_window), np.argsort(y_window), rowvar=False)[0, 1]
            return pd.Series(result, index=x.index)

        factors = factors.groupby(['code']).apply(rolling_spearmanr).droplevel(0)
        factors_mean = factors.groupby(['code']).rolling(windows).mean().droplevel(0)
        return factors, factors_mean


class QIML0224:

    @staticmethod
    def rankcorr(x):
        x_copy = x.copy()
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_copy['volume'].values.reshape(-1, 1))
        kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
        x_copy['volume_cluster'] = kmeans.fit_predict(x_scaled)

        volume_list = []
        for cluster_num in range(3):
            cluster_data = x_copy[x_copy['volume_cluster'] == cluster_num]['volume'].mean()
            volume_list.append(cluster_data)
            x_copy.loc[x_copy['volume_cluster'] == cluster_num, 'volume_cluster'] = cluster_data

        x_copy.loc[x_copy['volume_cluster'] == max(volume_list), 'volume_cluster'] = 'volume_cluster_big'
        x_copy.loc[x_copy['volume_cluster'] == min(volume_list), 'volume_cluster'] = 'volume_cluster_small'

        small_volume = x_copy[x_copy['volume_cluster'] == 'volume_cluster_small']['volume'].sum()
        ret = x_copy.close[-1] / x_copy.open[0] - 1
        return small_volume, ret
    
    

    @staticmethod
    def calculate(df, frequency='D', windows=24):
        factors = df.groupby(['code']).resample(frequency).apply(QIML0224.rankcorr)
        factors = pd.DataFrame(factors.tolist(), index=factors.index)
        factors.columns = ['small_volume', 'ret']

        def rolling_spearmanr(x):
            result = np.empty(len(x['small_volume']), dtype=float)
            result.fill(np.nan)

            for i in range(windows - 1, len(x['small_volume'])-1):
                x_window = x['small_volume'][i - windows + 2: i + 2]
                y_window = x['ret'][i - windows + 1: i + 1]
                result[i] = np.corrcoef(np.argsort(x_window), np.argsort(y_window), rowvar=False)[0, 1]
            return pd.Series(result, index=x.index)

        factors = factors.groupby(['code']).apply(rolling_spearmanr).droplevel(0)
        factors_mean = factors.groupby(['code']).rolling(windows).mean().droplevel(0)
        return factors, factors_mean



class QIML0330:
    """大单买入占比"""
    @staticmethod
    def big_buy(x):
        x_mean = x.buy_volume.mean()
        x_std = x.buy_volume.std()
        x_adj = x[['buy_volume', 'volume']][x.buy_volume > x_mean + x_std]
        factor = x_adj.buy_volume.sum() / x_adj.volume.sum()
        return factor
        
    @staticmethod
    def calculate(df, frequency='D', windows=20):
        factors = df.groupby(['code']).resample(frequency).apply(QIML0330.big_buy)
        factors_mean = factors.groupby(['code']).rolling(windows).mean().droplevel(0)
        return factors, factors_mean



class QIML0417:
    """买单集中度"""
    @staticmethod
    def buy(x):
        factor = (x.buy_volume.sum()) ** 2 / (x.volume.sum()) ** 2
        return factor
        
    @staticmethod
    def calculate(df, frequency='D', windows=20):
        factors = df.groupby(['code']).resample(frequency).apply(QIML0417.buy)
        factors_mean = factors.groupby(['code']).rolling(windows).sum().droplevel(0)
        return factors, factors_mean





class QIML0419:
    """开盘后大单净买入强度"""
    @staticmethod
    def big_diff(x):
        x_mean = x.volume.mean()
        x_std = x.volume.std()
        x_adj = x[x.volume > x_mean + x_std]
        x_diff = x_adj['buy_volume'] - x_adj['sell_volume']
        factor = x_diff.mean() / x_diff.std()
        return factor
        
    @staticmethod
    def calculate(df, frequency='D', windows=20):
        factors = df.groupby(['code']).resample(frequency).apply(QIML0419.big_diff)
        factors_mean = factors.groupby(['code']).rolling(windows).mean().droplevel(0)
        return factors, factors_mean



class QIML0425:
    """开盘后大单净买入占比"""
    @staticmethod
    def big_diff(x):
        x_mean = x.volume.mean()
        x_std = x.volume.std()
        x_adj = x[x.volume > x_mean + x_std]
        x_diff = x_adj['buy_volume'] - x_adj['sell_volume']
        factor = x_diff.sum() / x_adj.volume.sum()
        return factor
        
    @staticmethod
    def calculate(df, frequency='D', windows=20):
        factors = df.groupby(['code']).resample(frequency).apply(QIML0425.big_diff)
        factors_mean = factors.groupby(['code']).rolling(windows).mean().droplevel(0)
        return factors, factors_mean



class QIML0502:
    """主买占比"""
    @staticmethod
    def big_diff(x):
        factor = x.buy_volume.sum()/ x.volume.sum()
        return factor
        
    @staticmethod
    def calculate(df, frequency='D', windows=20):
        factors = df.groupby(['code']).resample(frequency).apply(QIML0502.big_diff)
        factors_mean = factors.groupby(['code']).rolling(windows).mean().droplevel(0)
        return factors, factors_mean


class QIML0508:
    """大单买入强度"""
    @staticmethod
    def big_diff(x):
        x_mean = x.buy_volume.mean()
        x_std = x.buy_volume.std()
        x_adj = x[x.buy_volume > x_mean + x_std]
        factor = x_adj.buy_volume.mean() / x_adj.buy_volume.std()
        return factor
        
    @staticmethod
    def calculate(df, frequency='D', windows=20):
        factors = df.groupby(['code']).resample(frequency).apply(QIML0508.big_diff)
        factors_mean = factors.groupby(['code']).rolling(windows).mean().droplevel(0)
        return factors, factors_mean


class QIML0518:
    """交易量变异系数"""
    @staticmethod
    def vcv(x):
        factor = x.volume.mean() / x.volume.std()
        return factor
    
    @staticmethod    
    def calculate(df, frequency='D', windows=20):
        factors = df.groupby(['code']).resample(frequency).apply(QIML0518.vcv)
        factors_mean = factors.groupby(['code']).rolling(windows).mean().droplevel(0)
        return factors, factors_mean


class QIML0525:
    """开盘后净主买占比"""
    
    @staticmethod 
    def net_buy(x):
        net = x.buy_volume - x.sell_volume
        factor = net.sum() / x.volume.sum()
        return factor

    @staticmethod    
    def calculate(df, frequency='D', windows=20):
        factors = df.groupby(['code']).resample(frequency).apply(QIML0525.net_buy)
        factors_mean = factors.groupby(['code']).rolling(windows).mean().droplevel(0)
        return factors, factors_mean

class QIML0602:
    """主买强度"""
    @staticmethod   
    def buy(x):
        factor = x.buy_volume.mean() / x.buy_volume.std()
        return factor

    @staticmethod    
    def calculate(df, frequency='D', windows=20):
        factors = df.groupby(['code']).resample(frequency).apply(QIML0602.buy)
        factors_mean = factors.groupby(['code']).rolling(windows).mean().droplevel(0)
        return factors, factors_mean

class QIML0612:
    """开盘后净主买强度"""
    @staticmethod 
    def net_buy(x):
        net = x.buy_volume - x.sell_volume
        factor = net.mean() / net.std()
        return factor
    
    
    @staticmethod    
    def calculate(df, frequency='D', windows=20):
        factors = df.groupby(['code']).resample(frequency).apply(QIML0612.net_buy)
        factors_mean = factors.groupby(['code']).rolling(windows).mean().droplevel(0)
        return factors, factors_mean


class QIML0908:
    """T分布主动占比"""
    """第一个是NAN 有失真"""
    @staticmethod 
    def t_buy(x):
        ret = x.close.pct_change()
        df = 1
        t_amount = x.volume * st.t.cdf(ret/ret.std(), df)
        factor = t_amount.sum() / x.volume.sum()
        return factor
    
    @staticmethod    
    def calculate(df, frequency='D', windows=20):
        factors = df.groupby(['code']).resample(frequency).apply(QIML0908.t_buy)
        factors_mean = factors.groupby(['code']).rolling(windows).mean().droplevel(0)
        return factors, factors_mean


class QIML0917:
    """置信正态分布主动占比"""
    """第一个是NAN 有失真"""
    @staticmethod 
    def n_buy(x):
        ret = x.close.pct_change()
        n_amount = x.volume * st.norm.cdf(ret/0.1 * 1.96)
        factor = n_amount.sum() / x.volume.sum()
        return factor
    
    @staticmethod    
    def calculate(df, frequency='D', windows=20):
        factors = df.groupby(['code']).resample(frequency).apply(QIML0917.n_buy)
        factors_mean = factors.groupby(['code']).rolling(windows).mean().droplevel(0)
        return factors, factors_mean



class QIML0923:
    """朴素主动占比"""
    """第一个是NAN 有失真"""
    @staticmethod
    def s_buy(x):
        close_diff = x.close.diff()
        df = 1
        s_amount = x.volume * st.t.cdf(close_diff/x.close.std(), df)
        factor = s_amount.sum() / x.volume.sum()
        return factor
    
        @staticmethod    
    def calculate(df, frequency='D', windows=20):
        factors = df.groupby(['code']).resample(frequency).apply(QIML0923.s_buy)
        factors_mean = factors.groupby(['code']).rolling(windows).mean().droplevel(0)
        return factors, factors_mean


class QIML1020:
    """均匀分布主动占比"""
    """第一个是NAN 有失真"""
    """币上不均匀"""
    @staticmethod 
    def m_buy(x):
        ret = x.close.pct_change()
        m_amount = x.volume * (ret - 0.1) / 0.2
        factor = m_amount.sum() / x.volume.sum()
        return factor
    @staticmethod    
    def calculate(df, frequency='D', windows=20):
        factors = df.groupby(['code']).resample(frequency).apply(QIML1020.m_buy)
        factors_mean = factors.groupby(['code']).rolling(windows).mean().droplevel(0)
        return factors, factors_mean



class QIML1103:
    """剔除超大单后的普通大单买入占比"""
    @staticmethod 
    def big_buy(x): 
        buy_up = x.buy_volume.quantile(0.9)
        buy_down = x.buy_volume.quantile(0.7)
        x_buy = x[(x.buy_volume >= buy_down) & (x.buy_volume < buy_up)]
        sell_up = x.sell_volume.quantile(0.9)
        sell_down = x.sell_volume.quantile(0.7)
        x_sell = x[(x.sell_volume >= sell_down) & (x.sell_volume < sell_up)]
        factor = x_buy.buy_volume.sum() / (x_buy.buy_volume.sum() + x_sell.sell_volume.sum())
        return factor
    
    @staticmethod    
    def calculate(df, frequency='D', windows=20):
        factors = df.groupby(['code']).resample(frequency).apply(QIML1103.big_buy)
        factors_mean = factors.groupby(['code']).rolling(windows).mean().droplevel(0)
        return factors, factors_mean

class QIML1119:
    """超大单买入占比"""
    @staticmethod  
    def super_buy(x): 
        buy_up = x.buy_volume.quantile(0.9)
        x_buy = x[x.buy_volume >= buy_up]
        sell_up = x.sell_volume.quantile(0.9)
        x_sell = x[x.sell_volume >= sell_up]
        factor = x_buy.buy_volume.sum() / (x_buy.buy_volume.sum() + x_sell.sell_volume.sum())
        return factor
    
    @staticmethod    
    def calculate(df, frequency='D', windows=20):
        factors = df.groupby(['code']).resample(frequency).apply(QIML1119.super_buy)
        factors_mean = factors.groupby(['code']).rolling(windows).mean().droplevel(0)
        return factors, factors_mean

class QIML1129:
    """小买单主动成交度"""
    @staticmethod 
    def small_buy(x):
        x_mean = x.buy_volume.mean()
        x_adj = x[['buy_volume', 'volume']][x.buy_volume < x_mean]
        factor = x_adj.buy_volume.sum() / x_adj.volume.sum()
        return factor
    
    @staticmethod    
    def calculate(df, frequency='D', windows=20):
        factors = df.groupby(['code']).resample(frequency).apply(QIML1129.small_buy)
        factors_mean = factors.groupby(['code']).rolling(windows).mean().droplevel(0)
        return factors, factors_mean


















