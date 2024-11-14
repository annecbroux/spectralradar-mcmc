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

path_root_list_baseline = [
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt/mcmc_chains_spectra_edr1e-4_cov_discard_noise_sigma0.35_freeturb_pt_1lev_swap1/',
                #   '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt/mcmc_chains_spectra_edr1e-4_cov_discard_noise_sigma0.35_freeturb_pt_5lev_swap1/',
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt/mcmc_chains_spectra_edr1e-4_cov_discard_noise_sigma0.35_freeturb_pt_1lev_swap1_no_parallel/',
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt_change_ssrg/mcmc_chains_spectra_edr1e-4_cov_discard_noise_sigma0.35_freeturb_pt_1lev_swap1_ctrl/'
                  ]

path_root_list_wind1 = [
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt/mcmc_chains_spectra_edr1e-4_cov_discard_noise_sigma0.35_freeturb_changed_wind_w_pt_1lev_swap1/',
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt/mcmc_chains_spectra_edr1e-4_cov_discard_noise_sigma0.35_freeturb_changed_wind_w_dw0.1_pt_1lev_swap1/',
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt/mcmc_chains_spectra_edr1e-4_cov_discard_noise_sigma0.35_freeturb_changed_wind_w_dw0.25_pt_1lev_swap1/',
                #   '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt/mcmc_chains_spectra_edr1e-4_cov_discard_noise_sigma0.35_freeturb_changed_wind_w_pt_5lev_swap1/',
                  ]

# path_root_list_wind_dw = [
                #   '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt/mcmc_chains_spectra_edr1e-4_cov_discard_noise_sigma0.35_freeturb_changed_wind_w_dw0.25_pt_1lev_swap1/',
# ]

path_root_list_wind2 = [
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt/mcmc_chains_spectra_edr1e-4_cov_discard_noise_sigma0.35_freeturb_changed_wind_w2_dw0.25_pt_1lev_swap1/', 
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt/mcmc_chains_spectra_edr1e-4_cov_discard_noise_sigma0.35_freeturb_changed_wind2_w_pt_1lev_swap1/' ,
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt/mcmc_chains_spectra_edr1e-4_cov_discard_noise_sigma0.35_freeturb_changed_wind2_w_std3_pt_1lev_swap1/', 
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt/mcmc_chains_spectra_edr1e-4_cov_discard_noise_sigma0.35_freeturb_changed_wind2_w_std4_pt_1lev_swap1/' 
                  ]


path_root_list_wind0_dw1 = [
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt/mcmc_chains_spectra_edr1e-4_cov_discard_noise_sigma0.35_freeturb_changed_wind_w0_dw0.1_pt_1lev_swap1/' 
                  ]

path_root_list_wind0_dw2 = [
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt/mcmc_chains_spectra_edr1e-4_cov_discard_noise_sigma0.35_freeturb_changed_wind_w0_dw0.25_pt_1lev_swap1/' 
                  ]

label_list = ['baseline','wind 1', 'wind 2 ', 'wind dw0.1', 'wind dw0.25']
color_list = ['blue', 'green', 'magenta', 'orange','yellow']

fig_prefix = 'hist_changed_wind2_after_burnin_'
after_burnin = 5000

vars_to_plot = {'b_mass_size':{'index':3,'title':'b$_m$ [-]','log_scale':False},
                'Deff':{'index':0,'title':'D$_{eff}$ [m]','log_scale':False},
                # 'iwc':{'index':7,'title':'IWC [kg m$^{-3}$]','log_scale':[False,True]},
                'logM0':{'index':1,'title':'log(N$_T$)','log_scale':False},
                'mu':{'index':2,'title':r'$\mu$ [-]','log_scale':False},
                'beta_area_size':{'index':4,'title':r'$\beta_a$ [-]','log_scale':False},
                # 'logalpha_area_size':{'index':5,'title':r'$\log(\alpha_a})$','log_scale':False},
                # 'loga_mass_size':{'index':6,'title':r'$\log(a_m)$','log_scale':False},
                }


nvar = len(vars_to_plot)


for target_spec in range(2,3):
    print(target_spec)
    hist_list_baseline = []  
    for i, path_root in enumerate(path_root_list_baseline):
        hist_list_baseline.append(compute_hist(path_root, target_spec, verbose=0,after_burnin=after_burnin)[0])
    hist_list_baseline = pd.concat(hist_list_baseline)
    if len(hist_list_baseline)==0:
        continue

    hist_list_wind1 = []  
    for i, path_root in enumerate(path_root_list_wind1):
        hist_list_wind1.append(compute_hist(path_root, target_spec, verbose=0,after_burnin=after_burnin)[0])
    hist_list_wind1 = pd.concat(hist_list_wind1)

    hist_list_wind2 = []  
    for i, path_root in enumerate(path_root_list_wind2):
        hist_list_wind2.append(compute_hist(path_root, target_spec, verbose=0,after_burnin=after_burnin)[0])
    hist_list_wind2 = pd.concat(hist_list_wind2)

    hist_list_wind0_dw1 = []  
    for i, path_root in enumerate(path_root_list_wind0_dw1):
        hist_list_wind0_dw1.append(compute_hist(path_root, target_spec, verbose=0,after_burnin=after_burnin)[0])
    hist_list_wind0_dw1 = pd.concat(hist_list_wind0_dw1)

    hist_list_wind0_dw2 = []  
    for i, path_root in enumerate(path_root_list_wind0_dw2):
        hist_list_wind0_dw2.append(compute_hist(path_root, target_spec, verbose=0, after_burnin=after_burnin)[0])
    hist_list_wind0_dw2 = pd.concat(hist_list_wind0_dw2)

    ground_truth = compute_hist(path_root_list_baseline[0], target_spec, verbose=0,after_burnin=after_burnin)[1]

    hist_list_baseline['Deff'] = 10**hist_list_baseline['logDeff']
    hist_list_baseline['wind_w'] = 0
    hist_list_baseline['dw_X'] = 0
    hist_list_baseline['dw_Ka'] = 0

    hist_list_wind1['Deff'] = 10**hist_list_wind1['logDeff']
    hist_list_wind1['dw_X'] = 0
    hist_list_wind1['dw_Ka'] = 0

    hist_list_wind2['Deff'] = 10**hist_list_wind2['logDeff']
    hist_list_wind2['dw_X'] = 0
    hist_list_wind2['dw_Ka'] = 0

    hist_list_wind0_dw1['Deff'] = 10**hist_list_wind0_dw1['logDeff']
    hist_list_wind0_dw2['Deff'] = 10**hist_list_wind0_dw2['logDeff']


    hist_list_joint = pd.concat((hist_list_baseline, hist_list_wind1,hist_list_wind2, hist_list_wind0_dw1,hist_list_wind0_dw2)).reset_index(drop=True)

    fig, axs = plt.subplots(1,nvar,figsize=(3*nvar,5))
    for var in vars_to_plot.keys():
        i_v = vars_to_plot[var]['index']
        d=1
        # bins_bs = max(1, int(len(hist_list_baseline[var])**(1/3))) #int(np.sqrt(len(hist_list_free_turb[iedr])))+1
        # bins_w1 = max(1, int(len(hist_list_wind1[var])**(1/3))) #int(np.sqrt(len(hist_list_free_turb[iedr])))+1
        # bins_w2 = max(1, int(len(hist_list_wind2[var])**(1/3))) #int(np.sqrt(len(hist_list_free_turb[iedr])))+1
        # bins_w0dw1 = max(1, int(len(hist_list_wind0_dw1[var])**(1/3))) #int(np.sqrt(len(hist_list_free_turb[iedr])))+1
        # bins_w0dw2 = max(1, int(len(hist_list_wind0_dw2[var])**(1/3))) #int(np.sqrt(len(hist_list_free_turb[iedr])))+1
        bins = 100
        if var=='iwc':
            bins = np.logspace(np.log10(np.nanpercentile(hist_list_joint['iwc'][hist_list_joint['iwc']>0],.2)),np.log10(np.nanpercentile(hist_list_joint['iwc'][hist_list_joint['iwc']>0],99.9)),100)
            d = 0
            if 'horiz' in fig_prefix:
                axs[i_v].set_xscale('log')
            else:
                axs[i_v].set_yscale('log')

        elif var =='Deff':
            # bins_bs = np.linspace(0,max(0.006,np.nanpercentile(hist_list_joint['Deff'],95)),bins_bs)
            # bins_w1 = np.linspace(0,max(0.006,np.nanpercentile(hist_list_joint['Deff'],95)),bins_w1)
            # bins_w2 = np.linspace(0,max(0.006,np.nanpercentile(hist_list_joint['Deff'],95)),bins_w2)
            # bins_w0dw1 = np.linspace(0,max(0.006,np.nanpercentile(hist_list_joint['Deff'],95)),bins_w0dw1)
            # bins_w0dw2 = np.linspace(0,max(0.006,np.nanpercentile(hist_list_joint['Deff'],95)),bins_w0dw2)
            bins = np.linspace(0,np.nanpercentile(hist_list_joint['Deff'],99.5),bins)
        else:
            bins = np.linspace(np.nanmin(hist_list_joint[var]),np.nanmax(hist_list_joint[var]),bins)

        
        nbs,bbs = np.histogram(hist_list_baseline[var], bins=bins,density=d)
        axs[i_v].step(-nbs,bbs[1:],linewidth=2.5,color='blue')
        ymin = -np.nanmax(nbs)

        nwind1, bwind1 = np.histogram(hist_list_wind1[var], bins=bins,density=d)
        ymax_wind1 = np.nanmax(nwind1)
        axs[i_v].step(nwind1,bwind1[1:],linewidth=2.5,color='green')

        nwind2, bwind2 = np.histogram(hist_list_wind2[var], bins=bins,density=d)
        if len(nwind2)>0:
            ymax_wind2 = np.nanmax(nwind2)
        else:
            ymax_wind2 = 0
        axs[i_v].step(nwind2,bwind2[1:],linewidth=2.5,color='magenta')

        nwind0_dw1, bwind0_dw1 = np.histogram(hist_list_wind0_dw1[var], bins=bins,density=d)
        if len(nwind0_dw1)>0:
            ymax_wind0_dw1 = np.nanmax(nwind0_dw1)
        else:
            ymax_wind0_dw1 = 0
        axs[i_v].step(nwind0_dw1,bwind0_dw1[1:],linewidth=2.5,color='orange')

        nwind0_dw2, bwind0_dw2 = np.histogram(hist_list_wind0_dw2[var], bins=bins,density=d)
        if len(nwind0_dw2)>0:
            ymax_wind0_dw2 = np.nanmax(nwind0_dw2)
        else:
            ymax_wind0_dw2 = 0
        axs[i_v].step(nwind0_dw2,bwind0_dw2[1:],linewidth=2.5,color='yellow')


        ymax = max(ymax_wind1,ymax_wind2,ymax_wind0_dw1,ymax_wind0_dw2)
        xmax = max(bbs.max(),bwind1.max(), bwind2.max(),bwind0_dw1.max(),bwind0_dw2.max())
        xmin = min(bbs.min(),bwind1.min(), bwind2.min(),bwind0_dw1.min(),bwind0_dw2.min())

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
    ]
    axs[-1].legend(
        custom_lines, 
        label_list, 
        # title="EDR",
        bbox_to_anchor=(1, 1),
        loc='upper left',
        fontsize=13
    )
    fig.savefig('figures/'+fig_prefix+'_spec%02d'%target_spec,bbox_inches='tight',facecolor='w',dpi=300)