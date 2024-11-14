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
from summary_tools import compute_hist

from scipy import special
plt.rcParams.update({'font.size': 14})

###################################################3

fig_prefix = 'hist_turb_free_vs_fixed_after_burnin'

path_root_list_true = [
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt/mcmc_chains_spectra_edr1e-4_cov_discard_noise_sigma0.35_freeturb_pt_1lev_swap1/',
                #   '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt/mcmc_chains_spectra_edr1e-4_cov_discard_noise_sigma0.35_freeturb_pt_5lev_swap1/',
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt/mcmc_chains_spectra_edr1e-4_cov_discard_noise_sigma0.35_freeturb_pt_1lev_swap1_no_parallel/',
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt_change_ssrg/mcmc_chains_spectra_edr1e-4_cov_discard_noise_sigma0.35_freeturb_pt_1lev_swap1_ctrl/'
]

path_root_list_free_turb = [
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt/mcmc_chains_spectra_edr5e-3_cov_discard_noise_sigma0.35_freeturb_pt_1lev_swap1/',
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt/mcmc_chains_spectra_edr1e-3_cov_discard_noise_sigma0.35_freeturb_pt_1lev_swap1/',
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt/mcmc_chains_spectra_edr5e-4_cov_discard_noise_sigma0.35_freeturb_pt_1lev_swap1/',
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt/mcmc_chains_spectra_edr1e-4_cov_discard_noise_sigma0.35_freeturb_pt_1lev_swap1/',
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt/mcmc_chains_spectra_edr5e-5_cov_discard_noise_sigma0.35_freeturb_pt_1lev_swap1/',
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt/mcmc_chains_spectra_edr1e-5_cov_discard_noise_sigma0.35_freeturb_pt_1lev_swap1/',


                  ]

path_root_list_fixed_turb = [
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt/mcmc_chains_spectra_edr5e-3_cov_discard_noise_sigma0.35_pt_1lev_swap1/',
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt/mcmc_chains_spectra_edr1e-3_cov_discard_noise_sigma0.35_pt_1lev_swap1/',
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt/mcmc_chains_spectra_edr5e-4_cov_discard_noise_sigma0.35_pt_1lev_swap1/',
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt/mcmc_chains_spectra_edr1e-4_cov_discard_noise_sigma0.35_pt_1lev_swap1/',
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt/mcmc_chains_spectra_edr5e-5_cov_discard_noise_sigma0.35_pt_1lev_swap1/',
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt/mcmc_chains_spectra_edr1e-5_cov_discard_noise_sigma0.35_pt_1lev_swap1/'
                  ]

# label_list = ['turb 1e-5', 'turb 5e-5', 'turb 1e-4', 'turb 5e-4','turb 1e-3', 'turb 5e-3']
label_list = ['turb 5e-3','turb 1e-3', 'turb 5e-4', 'turb 1e-4', 'turb 5e-5', 'turb 1e-5']
edr_list = [5e-3, 1e-3,5e-4,1e-4, 5e-5, 1e-5]
# color_list = ['red', 'orange', 'yellow','green', 'blue','darkviolet']
color_list = ['darkviolet','blue', 'green', 'yellow', 'orange', 'red', 'pink']

vars_to_plot = {'b_mass_size':{'index':3,'title':'b$_{mass-size}$ [-]','log_scale':False},
                'Deff':{'index':0,'title':'D$_{eff}$ [m]','log_scale':False},
                # 'iwc':{'index':5,'title':'IWC [kg m$^{-3}$]','log_scale':[False,True]},
                'logM0':{'index':1,'title':'log(N$_T$)','log_scale':False},
                'mu':{'index':2,'title':r'$\mu$ [-]','log_scale':False},
                'beta_area_size':{'index':4,'title':r'$\beta_{area-size}$ [-]','log_scale':False}
                }

after_burnin = 5000
nvar = len(vars_to_plot)

target_spec_list = range(5,6)


######################################################

for target_spec in target_spec_list:
    print(target_spec)
    hist_list_true = []
    for i, path_root in enumerate(path_root_list_true):
        hist_list_true.append(compute_hist(path_root, target_spec, verbose=0,after_burnin=after_burnin)[0])
    if sum([len(hist_list_true[ih]) for ih in range(len(hist_list_true))])==0:
        continue
    hist_list_true = pd.concat(hist_list_true).reset_index(drop=True)

    hist_list_free_turb = []
    for i, path_root in enumerate(path_root_list_free_turb):
        hist_list_free_turb.append(compute_hist(path_root, target_spec, verbose=0,after_burnin=after_burnin)[0])
    if sum([len(hist_list_free_turb[ih]) for ih in range(len(hist_list_free_turb))])==0:
        continue

    hist_list_fixed_turb = []
    for i, path_root in enumerate(path_root_list_fixed_turb):
        hist_list_fixed_turb.append(compute_hist(path_root, target_spec, verbose=0,after_burnin=after_burnin)[0])

    ground_truth = compute_hist(path_root_list_free_turb[0], target_spec, verbose=0,after_burnin=after_burnin)[1]

    hist_list_true['Deff'] = 10**hist_list_true['logDeff']

    for i in range(len(edr_list)):
        hist_list_free_turb[i]['Deff']=10**hist_list_free_turb[i]['logDeff']
        hist_list_fixed_turb[i]['Deff']=10**hist_list_fixed_turb[i]['logDeff']

    for i in range(len(hist_list_fixed_turb)):
        hist_list_free_turb[i]['free_turb']=np.ones(len(hist_list_free_turb[i]))*1
        hist_list_free_turb[i]['edr_true']=(edr_list[i])*np.ones(len(hist_list_free_turb[i]))
        hist_list_fixed_turb[i]['free_turb']=np.zeros(len(hist_list_fixed_turb[i]))
        hist_list_fixed_turb[i]['edr_true']=(edr_list[i])*np.ones(len(hist_list_fixed_turb[i]))

    hist_list_joint = pd.concat([hist_list_true]+hist_list_free_turb+hist_list_fixed_turb).reset_index(drop=True)
    hist_list_joint['free_turb_string']=hist_list_joint['free_turb'].map({1:'free',0:'fixed'})


    fig, axs = plt.subplots(1,nvar,figsize=(3*nvar,5))
    for var in vars_to_plot.keys():
        i_v = vars_to_plot[var]['index']
        # d=1
        ymax=-1
        ymin=1
        xmax=-1
        xmin=99999
        for iedr, edr in enumerate(edr_list):
            if edr ==1e-4:
                hist_list_free_turb[iedr] = hist_list_true

            # nbins_free = int(len(hist_list_free_turb[iedr])**(1/3))#int(np.sqrt(len(hist_list_free_turb[iedr])))+1
            # bins_free = nbins_free
            # nbins_fixed = int(len(hist_list_fixed_turb[iedr])**(1/3)) #int(np.sqrt(len(hist_list_fixed_turb[iedr])))+1
            # bins_fixed = nbins_fixed
            nbins_free = 100
            nbins_fixed = 100
            d=1
            if var=='iwc':
                bins_free = np.logspace(np.log10(np.nanpercentile(hist_list_free_turb[iedr]['iwc'][hist_list_free_turb[iedr]['iwc']>0],.1)),np.log10(np.nanpercentile(hist_list_free_turb[iedr]['iwc'][hist_list_free_turb[iedr]['iwc']>0],99)),nbins_free)
                bins_fixed = np.logspace(np.log10(np.nanpercentile(hist_list_fixed_turb[iedr]['iwc'][hist_list_fixed_turb[iedr]['iwc']>0],.1)),np.log10(np.nanpercentile(hist_list_fixed_turb[iedr]['iwc'][hist_list_fixed_turb[iedr]['iwc']>0],99)),nbins_fixed)
                d = 0
                axs[i_v].set_yscale('log')
            elif var =='Deff':
                bins_free = np.linspace(0,np.nanpercentile(hist_list_joint['Deff'], 99.9),nbins_free) #max(0.006,np.nanpercentile(hist_list_free_turb[iedr]['Deff'],93)),nbins_free)
                bins_fixed = np.linspace(0,np.nanpercentile(hist_list_joint['Deff'], 99.9),nbins_fixed) #max(0.006,np.nanpercentile(hist_list_free_turb[iedr]['Deff'],93)),nbins_free)
                # bins_fixed = np.linspace(0,max(0.006,np.nanpercentile(hist_list_fixed_turb[iedr]['Deff'],93)),nbins_fixed)
            else:
                bins_free = np.linspace(np.nanmin(hist_list_joint[var]),np.nanmax(hist_list_joint[var]),nbins_free)
                bins_fixed = np.linspace(np.nanmin(hist_list_joint[var]),np.nanmax(hist_list_joint[var]),nbins_fixed)
            nfree,bfree = np.histogram(hist_list_free_turb[iedr][var], bins=bins_free,density=d)
            ntrue,btrue = np.histogram(hist_list_fixed_turb[iedr][var], bins=bins_fixed,density=d)
            axs[i_v].step(ntrue,btrue[1:],linewidth=2.5,color=color_list[iedr],linestyle='--')
            axs[i_v].step(-nfree,bfree[1:],linewidth=2.5,color=color_list[iedr])

            ymax = max(ntrue.max(), ymax)
            ymin = min(-nfree.max(),ymin)

            xmax = max(bfree.max(),btrue.max(),xmax)
            xmin = min(bfree.min(),btrue.min(),xmin)

        axs[i_v].plot([ymin*1.1,ymax*1.1],[ground_truth[var],ground_truth[var]],'k--',lw=2)
        axs[i_v].set_xticks([])
        axs[i_v].fill_between([0,ymax*1.1],xmin,xmax,color='gray',alpha=.2)
        axs[i_v].set_xlim(ymin*1.1,ymax*1.1)
        axs[i_v].set_ylim(xmin,xmax)
        axs[i_v].set_title(vars_to_plot[var]['title'])
    fig.tight_layout()

    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color=color_list[i], lw=2, alpha=1) for i in range(len(edr_list))
    ]+[Line2D([0], [0], color='k', lw=2, linestyle='--', alpha=1)]
    axs[-1].legend(
        custom_lines,
        ["{:.0e}".format(e) for e in edr_list]+['ground truth'],
        title="EDR",
        bbox_to_anchor=(1, 1),
        loc='upper left',
        fontsize=13
    )

    fig.savefig('figures/'+fig_prefix+'_spec%02d'%target_spec,bbox_inches='tight',facecolor='w',dpi=300)