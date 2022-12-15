from pdb import set_trace as st

import numpy as np
import pandas as pd
from datetime import datetime as dt
import matplotlib.pyplot as plt




nn = 1000000

pp = ['prod', 'opex', 'cogs', 'asp']

df = {
    'time': [dt(2022,3,22), dt(2022,6,22), dt(2022,9,22)],
    pp[0]: [234000, 289000, 347000],
    pp[1]: np.array([5.341, 4.893, 5.234])*1e6,
    pp[2]: [37000, 37000, 36700],
    pp[3]: [57000, 56000, 55000],
    }

df = pd.DataFrame(df)




def lognorm(mu, sig):
    return np.random.lognormal(np.log(mu), np.log(sig+1), nn)


def plt_ts(axs, ii, jj, col):    
    df.plot('time', col, ax=axs[ii,jj], ls='-', marker='o')
    axs[ii,jj].set_ylabel(col)


def plt_hst(axs, ii, jj, lab, dat):
    axs[ii,jj].hist(dat, 100, density=True)
    axs[ii,jj].set_xlabel(lab)
    

def main():
    cagr = {}
    cagr[pp[0]] = 1 + lognorm(.42, .10)
    cagr[pp[1]] = 1 + lognorm(.12, .05)
    cagr[pp[2]] = 1 - lognorm(.08, .03)
    cagr[pp[3]] = 1 - lognorm(.10, .03)

    pe = lognorm(25, .25)

    fig, axs = plt.subplots(2, 5)

    for ii, pi in enumerate(pp):
        plt_ts(axs, 0, ii, pi)
        plt_hst(axs, 1, ii, 'cagr_'+pi, cagr[pi])

    plt_hst(axs, 1, 4, 'pe', pe)
    plt.show()
    st()


if __name__ == '__main__':
    main()
