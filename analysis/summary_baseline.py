import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from matplotlib import colors
plt.rcParams.update({'font.size': 15})
import glob
import os
import json
import seaborn as sns
from summary_tools import compute_hist, iwc_num_integ

from scipy import special
plt.rcParams.update({'font.size': 14})



path_root_list = [
                '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt/mcmc_chains_spectra_edr1e-4_cov_discard_noise_sigma0.35_freeturb_pt_1lev_swap1/',
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt/mcmc_chains_spectra_edr1e-4_cov_discard_noise_sigma0.35_freeturb_pt_5lev_swap1/',
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt/mcmc_chains_spectra_edr1e-4_cov_discard_noise_sigma0.35_freeturb_pt_1lev_swap1_no_parallel/',
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt_change_ssrg/mcmc_chains_spectra_edr1e-4_cov_discard_noise_sigma0.35_freeturb_pt_1lev_swap1_ctrl/'
                ]

fig_prefix = 'hist_baseline_horiz_after_burnin_'

after_burnin = 5000

target_spec_list = range(2,3)

vars_to_plot = {'b_mass_size':{'index':3,'title':'b$_{mass-size}$ [-]','log_scale':False},
                'Deff':{'index':0,'title':'D$_{eff}$ [m]','log_scale':False},
                'iwc':{'index':5,'title':'IWC [kg m$^{-3}$]','log_scale':[False,True]},
                'logM0':{'index':1,'title':'log(N$_T$)','log_scale':False},
                'mu':{'index':2,'title':r'$\mu$ [-]','log_scale':False},
                'beta_area_size':{'index':4,'title':r'$\beta_{area-size}$ [-]','log_scale':False}
                }
nvar = len(vars_to_plot)


for target_spec in target_spec_list:

    print('target_spec',target_spec)

    hist_list= []  
    for i, path_root in enumerate(path_root_list):
        hist_list.append(compute_hist(path_root, target_spec, verbose=0,after_burnin=after_burnin)[0])

    if sum([len(hist_list[ih]) for ih in range(len(hist_list))])==0:
        continue

    ground_truth = compute_hist(path_root_list[0], target_spec, verbose=0,after_burnin=after_burnin)[1]

    hist_list = pd.concat(hist_list).reset_index(drop=True)
    hist_list['Deff'] = 10**hist_list['logDeff']

    fig, axs = plt.subplots(1,nvar,figsize=(3*nvar,3))

    for var in vars_to_plot.keys():
        i_v = vars_to_plot[var]['index']
        ymax=-1
        ymin=1
        xmax=-1
        xmin=99999
        nbins = int(len(hist_list)**(1/3))
        d=1
        if var == 'Deff':
            bins = np.linspace(0,0.018,nbins)
        elif var == 'iwc':
            bins = np.logspace(-7.5,-2,nbins)
            d=0
            if 'horiz' in fig_prefix:
                axs[i_v].set_xscale('log')
        elif var == 'logM0':
            bins = np.linspace(-.5,5,nbins)
        elif var == 'mu':
            bins = np.linspace(-4,12,nbins)
        elif var == 'b_mass_size':
            bins = np.linspace(0.8,3.3,nbins)
        elif var == 'beta_area_size':
            bins = np.linspace(1,2.75,nbins)
        


        nfree,bfree = np.histogram(hist_list[var], bins=bins,density=d)
        axs[i_v].step(bfree[1:],nfree,linewidth=2.5,color='r')
    
        ymax = np.nanmax(nfree.max(), ymax)
        ymin = 0

        xmax = max(bfree.max(),xmax)
        xmin = min(bfree.min(),xmin)

        if 'horiz' in fig_prefix:
            axs[i_v].plot([ground_truth[var],ground_truth[var]],[ymin*1.1,ymax*1.1],'k--',lw=2)
            axs[i_v].set_yticks([])
            axs[i_v].set_xticklabels([])
            axs[i_v].set_ylim(ymin*1.1,ymax*1.1)
            # axs[i_v].set_xlim(xmin,xmax)
        else:
            axs[i_v].plot([ymin*1.1,ymax*1.1],[ground_truth[var],ground_truth[var]],'k--',lw=2)
            axs[i_v].set_xticks([])
            axs[i_v].set_xlim(ymin*1.1,ymax*1.1)
            axs[i_v].set_ylim(xmin,xmax)
        axs[i_v].set_title(vars_to_plot[var]['title'])
    fig.tight_layout()


    fig.savefig('figures/'+fig_prefix+'_noaxis_spec%02d'%target_spec,bbox_inches='tight',facecolor='w',dpi=300)