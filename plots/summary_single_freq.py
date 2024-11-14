import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from matplotlib import colors
import seaborn as sns
plt.rcParams.update({'font.size': 15})
import glob
import os
import json
import seaborn as sns

from scipy import special
from summary_tools import compute_hist


fig_prefix = 'histograms_singlefreq_after_burnin_'

path_root_list_single_freq = [
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt_singlefreq/mcmc_chains_spectra_X_edr1e-4_cov_discard_noise_sigma0.35_freeturb_pt_1lev_swap1/',
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt_singlefreq/mcmc_chains_spectra_Ka_edr1e-4_cov_discard_noise_sigma0.35_freeturb_pt_1lev_swap1/',
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt_singlefreq/mcmc_chains_spectra_W_edr1e-4_cov_discard_noise_sigma0.35_freeturb_pt_1lev_swap1/'
                  ]

path_root_list_true = [
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt/mcmc_chains_spectra_edr1e-4_cov_discard_noise_sigma0.35_freeturb_pt_1lev_swap1/',
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt/mcmc_chains_spectra_edr1e-4_cov_discard_noise_sigma0.35_freeturb_pt_5lev_swap1/',
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt/mcmc_chains_spectra_edr1e-4_cov_discard_noise_sigma0.35_freeturb_pt_1lev_swap1_no_parallel/',
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt_change_ssrg/mcmc_chains_spectra_edr1e-4_cov_discard_noise_sigma0.35_freeturb_pt_1lev_swap1_ctrl/'
                  ]


label_list = ['X', 'Ka', 'W']

color_list = ['green', 'fuchsia','yellow', 'orange', 'red', 'pink']

vars_to_plot = {'b_mass_size':{'index':3,'title':'b$_{mass-size}$ [-]','log_scale':False},
                'Deff':{'index':0,'title':'D$_{eff}$ [m]','log_scale':False},
                # 'iwc':{'index':5,'title':'IWC [kg m$^{-3}$]','log_scale':[False,True]},
                'logM0':{'index':1,'title':'log(N$_T$)','log_scale':False},
                'mu':{'index':2,'title':r'$\mu$ [-]','log_scale':False},
                'beta_area_size':{'index':4,'title':r'$\beta_{area-size}$ [-]','log_scale':False}
                }
after_burnin = 5000

nvar = len(vars_to_plot)

target_spec_list = range(7,8)

#######################################

for target_spec in target_spec_list:
    print(target_spec)
    hist_list_true = []  
    for i, path_root in enumerate(path_root_list_true):
        hist_list_true.append(compute_hist(path_root, target_spec, verbose=0,after_burnin=after_burnin)[0])
    
    if sum([len(hist_list_true[ih]) for ih in range(len(hist_list_true))])==0:
        continue
    hist_list_true = pd.concat(hist_list_true).reset_index(drop=True)

    hist_list_1f= []  
    for i, path_root in enumerate(path_root_list_single_freq):
        hist_list_1f.append(compute_hist(path_root, target_spec, verbose=0,after_burnin=after_burnin)[0])

    ground_truth = compute_hist(path_root_list_true[0], target_spec, verbose=0,after_burnin=after_burnin)[1]


    for i in range(len(hist_list_1f)):
        hist_list_1f[i]['Deff'] = 10**hist_list_1f[i]['logDeff']
        hist_list_1f[i]['freqs'] = 'single'

    # for i in range(len(hist_list_true)):
    hist_list_true['Deff'] = 10**hist_list_true['logDeff']
    hist_list_true['freqs'] = 'XKaW'

    hist_list_joint = pd.concat(hist_list_1f+[hist_list_true]).reset_index(drop=True)

    fig, axs = plt.subplots(1,nvar,figsize=(3*nvar,5))
    for var in vars_to_plot.keys():
        i_v = vars_to_plot[var]['index']
        d=1
        bins = 100
        if var=='iwc':
            bins = np.logspace(np.log10(np.nanpercentile(hist_list_joint['iwc'][hist_list_joint['iwc']>0],.1)),np.log10(np.nanpercentile(hist_list_joint['iwc'][hist_list_joint['iwc']>0],99.9)),100)
            d = 0
            axs[i_v].set_yscale('log')
        elif var =='Deff':
            bins=np.linspace(0,np.nanpercentile(hist_list_joint['Deff'],99.9),100)
        else:
            bins = np.linspace(np.nanmin(hist_list_joint[var]),np.nanmax(hist_list_joint[var]),100)

        ymax = -1
        xmin = 9999
        xmax = -9999

        for i1f in range(len(hist_list_1f)):
            n1f,b1f = np.histogram(hist_list_1f[i1f][var], bins=bins,density=d)
            axs[i_v].step(n1f,b1f[1:],linewidth=2.5,color=color_list[i1f])
            ymax = max(n1f.max(),ymax)
            xmin = min(b1f.min(),xmin)
            xmax = max(b1f.max(),xmax)
            
        ntrue,btrue = np.histogram(hist_list_true[var], bins=bins,density=d)
        ymin = -ntrue.max()
        xmax = max(btrue.max(),xmax)
        xmin = min(btrue.min(),xmin)

        axs[i_v].step(-ntrue,btrue[1:],linewidth=2.5,color='blue')
        axs[i_v].plot([ymin*1.1,ymax*1.1],[ground_truth[var],ground_truth[var]],'k--',lw=2)
        axs[i_v].set_xticks([])
        axs[i_v].fill_between([0,ymax*1.1],xmin,xmax,color='gray',alpha=.2)
        axs[i_v].set_xlim(ymin*1.1,ymax*1.1)
        axs[i_v].set_ylim(xmin,xmax)
        axs[i_v].set_title(vars_to_plot[var]['title'])
    fig.tight_layout()
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color=color_list[i], lw=2, alpha=1) for i in range(len(label_list))
    ]+[Line2D([0], [0], color='b', lw=2, linestyle='-', alpha=1)]
    axs[-1].legend(
        custom_lines, 
        label_list+['reference'],
        bbox_to_anchor=(1, 1),
        loc='upper left',
        fontsize=13
    )
    fig.savefig('figures/'+fig_prefix+'_spec%02d'%target_spec,bbox_inches='tight',facecolor='w',dpi=300)