import os
from pdb import set_trace as st

import numpy as np
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta as td
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.stats as sts


def lognorm(mu, sig):
    return np.random.lognormal(np.log(mu), np.log(sig+1), nn)


def plt_ts(df, axs, ii, jj, col, cc):    
    df.plot('date', col, ax=axs[ii,jj], ls='-', marker='o', logy=True)

    df[col+'_ft'] = cc[0,jj]*cc[1,jj]**df.time
    df.plot('date', col+'_ft', ax=axs[ii,jj], ls='-', logy=True)
    
    sim[jj] = cc[0,jj] * cagr[col][np.newaxis,:] ** ftr.time.values[:,np.newaxis]
    ftr[col] = sim[jj].mean(axis=1)
    ftr[col+'-2'] = np.percentile(sim[jj], 100*sts.norm.cdf(-2), axis=1)
    ftr[col+'-1'] = np.percentile(sim[jj], 100*sts.norm.cdf(-1), axis=1)
    ftr[col+'+1'] = np.percentile(sim[jj], 100*sts.norm.cdf(+1), axis=1)
    ftr[col+'+2'] = np.percentile(sim[jj], 100*sts.norm.cdf(+2), axis=1)
    ftr.plot('date', col, ax=axs[ii,jj], ls='-', logy=True)
    axs[ii,jj].fill_between(ftr.date, ftr[col+'-2'], ftr[col+'+2'], color='tab:green', alpha=0.1)
    axs[ii,jj].fill_between(ftr.date, ftr[col+'-1'], ftr[col+'+1'], color='tab:green', alpha=0.2)
    
    axs[ii,jj].set_ylabel(col)
    axs[ii,jj].grid()
    axs[ii,jj].set_xlim((df.date.min(), ftr.date.max()))


def plt_hst(axs, ii, jj, lab, dat):
    axs[ii,jj].hist(dat, 100, density=True)
    axs[ii,jj].set_xlabel(lab)
    axs[ii,jj].grid()
    

def ff(tt, *par):
    par = np.array(par).reshape((2, len(pp)))
    # print(par)
    return np.log(par[0:1,:]*par[1:2,:]**tt).ravel()

    
def main():

    par = np.ones((2,len(pp)))
    low = np.zeros((2,len(pp))).ravel()
    hig = np.ones((2,len(pp))).ravel()*np.inf
    tt = df.time.values[:,np.newaxis]
    yy = np.log(df[pp].values).ravel()
    cc, cv = curve_fit(ff, tt, yy, p0=par.ravel(), bounds=(low, hig))
    cc = cc.reshape((2, len(pp)))
    
    fig, axs = plt.subplots(2, 5, figsize=(22,16))
    plt.subplots_adjust(bottom=0.01, right=0.99, top=0.95, left=0.04, wspace=0.25, hspace=0.25)

    for ii, pi in enumerate(pp):
        plt_ts(df, axs, 0, ii, pi, cc)
        plt_hst(axs, 1, ii, 'cagr_'+pi, cagr[pi])

    plt_hst(axs, 1, 4, 'pe', pe)

    
    rev = sim[0]*sim[3]
    cgs = sim[0]*sim[2]
    gprft = rev-cgs
    oprft = gprft-sim[1]
    mcap = oprft*pe*4

    col = 'mcap'
    ftr[col] = mcap.mean(axis=1)
    ftr[col+'-2'] = np.percentile(mcap, 100*sts.norm.cdf(-2), axis=1)
    ftr[col+'-1'] = np.percentile(mcap, 100*sts.norm.cdf(-1), axis=1)
    ftr[col+'+1'] = np.percentile(mcap, 100*sts.norm.cdf(+1), axis=1)
    ftr[col+'+2'] = np.percentile(mcap, 100*sts.norm.cdf(+2), axis=1)

    ax = axs[0,4]
    df.plot('date', 'mcap', ls='-', marker='o', logy=True, ax=ax)
    ftr.plot('date', col, ax=ax, ls='-', logy=True, c='#333333')
    ax.fill_between(ftr.date, ftr[col+'-2'], ftr[col+'+2'], color='tab:gray', alpha=0.1)
    ax.fill_between(ftr.date, ftr[col+'-1'], ftr[col+'+1'], color='tab:gray', alpha=0.2)
    ax.set_ylabel(col)
    ax.grid()
    ax.set_xlim((df.date.min(), ftr.date.max()))

    out = os.path.join('out', dt.today().strftime('%Y-%m-%d') + '.png')
    plt.savefig(out, bbox_inches='tight')

    fig = plt.figure(figsize=(5,5))
    plt.scatter(cagr[pp[2]], cagr[pp[3]], s=1)
    plt.xlabel('cagr_'+pp[2])
    plt.ylabel('cagr_'+pp[3])
    plt.grid()



################################################################
nn = 100000
diny = 365.26
yrs = 5
pp = ['prod', 'opex', 'cogs', 'asp']

df = {
    'date': [
        dt(2021,4,2), # Leave this as April 2nd and not April 1st. Pandas will format the date differently for the x-axis otherwise.
        dt(2021,7,1),
        dt(2021,10,1),
        dt(2022,1,1),
        dt(2022,4,1),
        dt(2022,7,1),
        dt(2022,10,1),
        dt(2023,1,1),
        dt(2023,4,1),
        dt(2023,7,1),
        dt(2023,10,1),
        dt(2024,1,1),
        dt(2024,4,1),
        dt(2024,7,1),
    ],
    # Total production
    pp[0]: [
        180338,
        206421,
        237823,
        305840,
        305407,
        258580,
        365923,
        439701,
        440808,
        479700,
        430488,
        494989,
        433371,
        410831,
    ],
    # Operating expenses
    pp[1]: np.array([
        1.621,
        1.572,
        1.656,
        1.894,
        1.857,
        1.770,
        1.694,
        1.876,
        1.847,
        2.134,
        2.414,
        2.374,
        2.525,
        2.973,
    ])*1e9,
    # Cost of revenues - Automotive sales
    pp[2]: np.array([
         6.617,
         7.307,
         8.150,
        10.689,
        10.914,
        10.153,
        13.099,
        15.433,
        15.422,
        16.841,
        15.656,
        17.202,
        13.897,
        15.962,
    ])*1e9,
    # Revenues - Automotive sales
    pp[3]: np.array([
         8.705,
         9.874,
        11.393,
        15.025,
        15.514,
        13.670,
        17.785,
        20.241,
        18.878,
        20.419,
        18.582,
        20.630,
        16.460,
        18.530,
    ])*1e9,
    # Market cap - Day after earnings
    'mcap': np.array([
        234.91,
        214.93,
        341.62,
        276.37,
        336.26,
        271.71,
        207.28,
        177.9,
        162.99,
        262.9,
        220.11,
        182.63,
        162.13,
        215.99,
    ])*3178921391
}

df = pd.DataFrame(df)
df.cogs /= df['prod']
df.asp /= df['prod']
df['time'] = (df.date - dt.today()).dt.days / diny

ftr = {
    'date': dt.today() + np.arange(0, yrs*4+1)*td(days=diny/4),
    'time': np.arange(0, yrs*4+1) / 4
}
ftr = pd.DataFrame(ftr)

sim = np.empty((len(pp), ftr.shape[0], nn))

cagr = {}
cagr[pp[0]] = 1 + lognorm(.20, .12)  # CAGR Production
cagr[pp[1]] = 1 + lognorm(.06, .30)  # CAGR OPEX

cov = np.array([[.2, 0.3],[0.3, 0.5]])  # CAGR Automotive COGS/ASP
tmp = np.exp(np.random.multivariate_normal([np.log(0.07), np.log(0.06)], np.log(cov+1), size=nn))
cagr[pp[2]] = 1 - tmp[:,0]
cagr[pp[3]] = 1 - tmp[:,1]
cagr[pp[3]] = np.where(cagr[pp[3]]<0.3, 0.3, cagr[pp[3]])

pe = lognorm(30, .2)



if __name__ == '__main__':
    main()
