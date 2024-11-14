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


####################################################################

path_root_list_free_ab = [
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt/mcmc_chains_spectra_edr1e-4_cov_discard_noise_sigma0.35_freeturb_pt_1lev_swap1/',
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt/mcmc_chains_spectra_edr1e-4_cov_discard_noise_sigma0.35_freeturb_pt_5lev_swap1/',
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt/mcmc_chains_spectra_edr1e-4_cov_discard_noise_sigma0.35_freeturb_pt_1lev_swap1_no_parallel/',
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt_change_ssrg/mcmc_chains_spectra_edr1e-4_cov_discard_noise_sigma0.35_freeturb_pt_1lev_swap1_ctrl/'
                  ]

path_root_list_ab_true = [
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt/mcmc_chains_spectra_edr1e-4_cov_discard_noise_sigma0.35_freeturb_pt_1lev_swap1_mutrue/',
                  ]

path_root_list_ab_bf = [
                  '/home/billault/Documents/In_progress/mcmc/tests_cv_longer_newdata/chains_spectra_pt/mcmc_chains_spectra_edr1e-4_cov_discard_noise_sigma0.35_freeturb_pt_1lev_swap1_mu0/',
                  ]

fig_prefix = 'histograms_mu_free_vs_fixed_after_burnin_'

color_list = ['blue','red','blue','green']

vars_to_plot = {'b_mass_size':{'index':3,'title':'b$_{mass-size}$ [-]','log_scale':False},
                'Deff':{'index':0,'title':'D$_{eff}$ [m]','log_scale':False},
                # 'iwc':{'index':5,'title':'IWC [kg m$^{-3}$]','log_scale':[False,True]},
                'logM0':{'index':1,'title':'log(N$_T$) [log(m$^{-3}$)]','log_scale':False},
                'mu':{'index':2,'title':r'$\mu$ [-]','log_scale':False},
                'beta_area_size':{'index':4,'title':r'$\beta_{area-size}$ [-]','log_scale':False}
                }

after_burnin = 5000

nvar = len(vars_to_plot)

target_spec_list = range(18,19)

####################################################################


for target_spec in target_spec_list:
    print(target_spec)
    hist_list_free_ab= []  
    for i, path_root in enumerate(path_root_list_free_ab):
        hist_list_free_ab.append(compute_hist(path_root, target_spec, verbose=0,after_burnin=after_burnin)[0])

    if sum([len(hist_list_free_ab[ih]) for ih in range(len(hist_list_free_ab))])==0:
        continue
    hist_list_free_ab = pd.concat(hist_list_free_ab).reset_index(drop=True)

    hist_list_ab_true = []  
    for i, path_root in enumerate(path_root_list_ab_true):
        hist_list_ab_true.append(compute_hist(path_root, target_spec, verbose=0,after_burnin=after_burnin)[0])
    hist_list_ab_true = pd.concat(hist_list_ab_true).reset_index(drop=True)

    hist_list_ab_bf = []  
    for i, path_root in enumerate(path_root_list_ab_bf):
        hist_list_ab_bf.append(compute_hist(path_root, target_spec, verbose=0,after_burnin=after_burnin)[0])
    hist_list_ab_bf = pd.concat(hist_list_ab_bf).reset_index(drop=True)

    ground_truth = compute_hist(path_root_list_free_ab[0], target_spec, verbose=0,after_burnin=after_burnin)[1]
    
    # for i in range(len(hist_list_free_ab)):
    hist_list_free_ab['free_ab']=np.ones(len(hist_list_free_ab))
    hist_list_free_ab['ab_true']=np.zeros(len(hist_list_free_ab))+0.5
    hist_list_ab_true['free_ab']=np.zeros(len(hist_list_ab_true))
    hist_list_ab_true['ab_true']=np.ones(len(hist_list_ab_true))
    hist_list_ab_bf['free_ab']=np.zeros(len(hist_list_ab_bf))
    hist_list_ab_bf['ab_true']=np.zeros(len(hist_list_ab_bf))
    hist_list_ab_bf['Deff'] = 10**hist_list_ab_bf['logDeff']
    hist_list_ab_true['Deff'] = 10**hist_list_ab_true['logDeff']
    hist_list_free_ab['Deff'] = 10**hist_list_free_ab['logDeff']

    hist_list_joint = pd.concat((hist_list_free_ab,hist_list_ab_true,hist_list_ab_bf)).reset_index(drop=True)
    hist_list_joint['free_ab_string']=hist_list_joint['free_ab'].map({0:'free',1:'fixed'})

    fig, axs = plt.subplots(1,nvar,figsize=(3*nvar,5))
    for var in vars_to_plot.keys():
        i_v = vars_to_plot[var]['index']
        d=1
        nbins = 75
        if var=='iwc':
            bins = np.logspace(np.log10(np.nanpercentile(hist_list_joint['iwc'][hist_list_joint['iwc']>0],1)),np.log10(np.nanpercentile(hist_list_joint['iwc'][hist_list_joint['iwc']>0],99.9)),nbins)
            d = 0
            axs[i_v].set_yscale('log')
        elif var =='Deff':
            # bins=np.linspace(0,.01,100)
            bins=np.linspace(0,np.nanpercentile(hist_list_joint['Deff'],99.5),nbins)
        else:
            bins=np.linspace(np.nanmin(hist_list_joint[var]),np.nanmax(hist_list_joint[var]),nbins)
            # bins=np.linspace(np.nanpercentile(hist_list_joint[var],0.01),np.nanpercentile(hist_list_joint[var],99.99),100)
        nfree,bfree = np.histogram(hist_list_free_ab[var], bins=bins,density=d)
        ntrue,btrue = np.histogram(hist_list_ab_true[var], bins=bins,density=d)
        nbf,bbf = np.histogram(hist_list_ab_bf[var], bins=bins,density=d)
        if var=='mu':
            ymax = nfree.max()
        else:
            ymax = max(ntrue.max(),nbf.max())
        ymin = -nfree.max()
        xmax = max(bfree.max(),btrue.max(),bbf.max())
        xmin = min(bfree.min(),btrue.min(),bbf.min())
        axs[i_v].step(ntrue,btrue[1:],linewidth=2.5,color='r')
        axs[i_v].step(nbf,bbf[1:],linewidth=2.5,color='green')
        axs[i_v].step(-nfree,bfree[1:],linewidth=2.5,color='blue')
        axs[i_v].plot([ymin*1.1,ymax*1.1],[ground_truth[var],ground_truth[var]],'k--',lw=2)
        axs[i_v].set_xticks([])
        axs[i_v].fill_between([0,ymax*1.1],xmin,xmax,color='gray',alpha=.2)
        axs[i_v].set_xlim(ymin*1.1,ymax*1.1)
        axs[i_v].set_ylim(xmin,xmax)
        axs[i_v].set_title(vars_to_plot[var]['title'])
        
    fig.tight_layout()
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color=color_list[i], lw=2, alpha=1) for i in [0,1,3]
    ]+[Line2D([0], [0], color='k', lw=2, linestyle='--', alpha=1)]
    axs[-1].legend(
        custom_lines, 
        [r'baseline', r'$\mu$_true','exp. ($\mu=0$)','ground truth'], 
        # title="EDR",
        bbox_to_anchor=(1, 1),
        loc='upper left',
        fontsize=13
    )
    fig.savefig('figures/'+fig_prefix+'_spec%02d'%target_spec,bbox_inches='tight',facecolor='w',dpi=300)