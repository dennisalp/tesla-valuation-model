from pdb import set_trace as st

import numpy as np
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta as td
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit



def lognorm(mu, sig):
    return np.random.lognormal(np.log(mu), np.log(sig+1), nn)


def plt_ts(df, axs, ii, jj, col, cc):    
    df.plot('date', col, ax=axs[ii,jj], ls='-', marker='o', logy=True)

    df[col+'_ft'] = cc[0,jj]*cc[1,jj]**df.time
    df.plot('date', col+'_ft', ax=axs[ii,jj], ls='-', logy=True)

    sim[jj] = cc[0,jj] * cagr[col][np.newaxis,:] ** ftr.time.values[:,np.newaxis]
    ftr[col] = sim[jj].mean(axis=1)
    ftr.plot('date', col, ax=axs[ii,jj], ls='-', logy=True)
    
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
    
    fig, axs = plt.subplots(2, 4)

    for ii, pi in enumerate(pp):
        plt_ts(df, axs, 0, ii, pi, cc)
        plt_hst(axs, 1, ii, 'cagr_'+pi, cagr[pi])

    # plt_hst(axs, 1, 4, 'pe', pe)
    plt.show()
    st()




################################################################
nn = 10000
diny = 365.26
yrs = 5
pp = ['prod', 'opex', 'cogs', 'asp']

df = {
    'date': [dt(2022,3,22), dt(2022,6,22), dt(2022,9,22)],
    pp[0]: [234000, 289000, 347000],
    pp[1]: np.array([5.341, 4.893, 5.234])*1e6,
    pp[2]: [37000, 37000, 36700],
    pp[3]: [57000, 56000, 55000],
    }

df = pd.DataFrame(df)
df['time'] = (df.date - dt.today()).dt.days / diny
ftr = {
    'date': dt.today() + np.arange(0, yrs*4+1)*td(days=diny/4),
    'time': np.arange(0, yrs*4+1) / 4
    }
ftr = pd.DataFrame(ftr)

sim = np.empty((len(pp), ftr.shape[0], nn))

cagr = {}
cagr[pp[0]] = 1 + lognorm(.42, .10)
cagr[pp[1]] = 1 + lognorm(.12, .05)
cagr[pp[2]] = 1 - lognorm(.08, .03)
cagr[pp[3]] = 1 - lognorm(.10, .03)

pe = lognorm(25, .25)



if __name__ == '__main__':
    main()
