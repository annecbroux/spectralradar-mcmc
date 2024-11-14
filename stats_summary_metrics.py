import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from summary_tools import compute_hist

# Set the directories

base_directory = '/home/billault/mcmc/chains/'
metrics_dir = '/home/billault/mcmc/metrics_after_burnin/'

# Configs on which to compute the metrics 
config_dict = {
    'baseline' : dict(
        path_root_list = [
                  base_directory+'mcmc_chains_spectra_edr1e-4_freeturb/',
                  ],
        prefix = 'baseline_'

    ), 

    'turbulence_1' : dict(
        path_root_list = [
                  base_directory+'mcmc_chains_spectra_edr5e-3_freeturb/'
                  ],
        prefix = 'freeturb_5e-3_'
    ),

    'turbulence_2' : dict(
        path_root_list = [
                  base_directory+'mcmc_chains_spectra_edr1e-5_freeturb/'
                  ],
        prefix = 'freeturb_1e-5_'
    ),

    'ab_true' : dict(
        path_root_list = [
                  base_directory+'mcmc_chains_spectra_edr1e-4_freeturb_abtrue/',
                ],
        prefix = 'ab_true_'
    ),

    'ab_bf' : dict(
        path_root_list = [
                        base_directory+'mcmc_chains_spectra_edr1e-4_freeturb_abbf/',
                        ],
        prefix = 'ab_bf_'
    ),

    'mu_true' : dict(
        path_root_list = [
                  base_directory+'mcmc_chains_spectra_edr1e-4_freeturb_mutrue/',
                ],
        prefix = 'mu_true_'
    ),

    'mu0' : dict(
        path_root_list = [
                        base_directory+'mcmc_chains_spectra_edr1e-4_freeturb_mu0/',
                        ],
        prefix = 'mu0_'
    ),

    'dual_freq_KaW' : dict(
        path_root_list = [
                  base_directory+'mcmc_chains_spectra_edr1e-4_freeturb_KaW/',
                ],
        prefix = 'dualfreq_KaW_'
    ),

    'dual_freq_XW' : dict(
        path_root_list = [
                  base_directory+'mcmc_chains_spectra_edr1e-4_freeturb_XW/',
                ],
        prefix = 'dualfreq_XW_'
    ),

    'dual_freq_XKa' : dict(
        path_root_list = [
                  base_directory+'mcmc_chains_spectra_edr1e-4_freeturb_XKa/',
                ],
        prefix = 'dualfreq_XKa_'
    ),

    'single_freq_W' : dict(
        path_root_list = [
                  base_directory+'mcmc_chains_spectra_edr1e-4_freeturb_W/',
                ],
        prefix = 'singlefreq_W_'
    ),

    'single_freq_Ka' : dict(
        path_root_list = [
                  base_directory+'mcmc_chains_spectra_edr1e-4_freeturb_Ka/',
                ],
        prefix = 'singlefreq_Ka_'
    ),

    'single_freq_X' : dict(
        path_root_list = [
                  base_directory+'mcmc_chains_spectra_edr1e-4_freeturb_X/',
                ],
        prefix = 'singlefreq_X_'
    ),

    'ssrg_hogan1' : dict(
        path_root_list = [
                        base_directory+'mcmc_chains_spectra_edr1e-4_freeturb_hogan1/'
                        ],
        prefix = 'ssrg_hogan1_'
    ),

    'ssrg_hogan2' : dict(
        path_root_list = [
                  base_directory+'mcmc_chains_spectra_edr1e-4_freeturb_hogan2/'
                ],
        prefix = 'ssrg_hogan2_'
    ),

    'ssrg_ori2' : dict(
        path_root_list = [
                        base_directory+'mcmc_chains_spectra_edr1e-4_freeturb_ori2/'
                        ],
        prefix = 'ssrg_ori2_'
    ),

    'ssrg_ori1' : dict(
        path_root_list = [
                        base_directory+'mcmc_chains_spectra_edr1e-4_freeturb_ori1/'
                        ],
        prefix = 'ssrg_ori1_'
    ),


    'wind1' : dict(
        path_root_list = [
                  base_directory+'mcmc_chains_spectra_edr1e-4_freeturb_wind1/',
                ],
        prefix = 'wind1_'
    ),


    'wind2' : dict(
        path_root_list = [
                  base_directory+'mcmc_chains_spectra_edr1e-4_freeturb_wind2/' ,
                ],
        prefix = 'wind2_'
    ),


    'wind0_dw0.1' : dict(
        path_root_list = [
                        base_directory+'mcmc_chains_spectra_edr1e-4_freeturb_wind_w0_dw0.1/'
                        ],
        prefix = 'wind_dw1_'
    ),

    'wind0_dw0.25' : dict(
        path_root_list = [
                        base_directory+'mcmc_chains_spectra_edr1e-4_freeturb_wind_w0_dw0.25/'
                        ],
        prefix = 'wind_dw2_'
    ),

    'ssrg_xw_ori1': dict(
        path_root_list = [
                  base_directory+'mcmc_chains_spectra_edr1e-4_freeturb_XW_ori1/'
                ],
        prefix = 'ssrg_xw_ori1_'
    ),

    'ssrg_xw_hogan2': dict(
        path_root_list = [
                  base_directory+'mcmc_chains_spectra_edr1e-4_freeturb_XW_hogan2/'
                ],
        prefix = 'ssrg_xw_hogan2_'
    ),

    'ssrg_xw_hogan1': dict(
        path_root_list = [
                  base_directory+'mcmc_chains_spectra_edr1e-4_freeturb_XW_hogan1/'
                ],
        prefix = 'ssrg_xw_hogan1_'
    ),

    'ssrg_kaw_ori1': dict(
        path_root_list = [
                  base_directory+'mcmc_chains_spectra_edr1e-4_freeturb_KaW_ori1/'
                ],
        prefix = 'ssrg_kaw_ori1_'
    ),

    'ssrg_kaw_hogan2': dict(
        path_root_list = [
                  base_directory+'mcmc_chains_spectra_edr1e-4_freeturb_KaW_hogan2/'
                ],
        prefix = 'ssrg_kaw_hogan2_'
    ),

    'ssrg_kaw_hogan1': dict(
        path_root_list = [
                  base_directory+'mcmc_chains_spectra_edr1e-4_freeturb_KaW_hogan1/'
                ],
        prefix = 'ssrg_kaw_hogan1_'
    )

}


# Additional parameters 
after_burnin = 5000 # Discard start of burnin period

# On which vars to compute metrics
vars_to_plot = [
                'b_mass_size',
                'logDeff',
                'Deff',
                'logM0',
                'M0',
                'mu',
                'beta_area_size',
]

nvar = len(vars_to_plot)


for config in config_dict:
    print(config)
    config = config_dict[config]
    df = {var:{'q10':[], 'q25':[], 'q50':[], 'q75':[], 'q90':[], 'mean':[],'std':[], 'skew':[], 'cv':[], 'qcd':[], 'std_2':[]} for var in vars_to_plot}
    for target_spec in range(2,19):
        print(target_spec)
        hist_list= []  
        dir_exist = 0
        for i, path_root in enumerate(config['path_root_list']):
            path_data = path_root+'targetspec%02d/'%target_spec
            if os.path.exists(path_data):
                dir_exist = 1
            hist_list.append(compute_hist(path_root, target_spec, verbose=0,after_burnin=after_burnin)[0])

        if dir_exist == 0:
            continue

        hist_list_joint = pd.concat(hist_list).reset_index(drop=True)
        hist_list_joint['Deff']=10**hist_list_joint['logDeff']
        hist_list_joint['M0']=10**hist_list_joint['logM0']
        hist_list_joint['edr']=10**hist_list_joint['logedr']
        hist_list_joint['a_mass_size']=10**hist_list_joint['loga_mass_size']
        hist_list_joint['alpha_area_size']=10**hist_list_joint['logalpha_area_size']
        hist_list_joint['logiwc']=np.log10(hist_list_joint['iwc'])

        if len(hist_list_joint)==0:
            for var in vars_to_plot:
                for key in df[var]:
                    df[var][key].append(-999)
        else:
            for var in vars_to_plot:
                [q10, q25, med, q75, q90] = np.nanquantile(hist_list_joint[var],[0.10,0.25,0.5, 0.75, 0.90])
                df[var]['q10'].append(q10)
                df[var]['q25'].append(q25)
                df[var]['q50'].append(med)
                df[var]['q75'].append(q75)
                df[var]['q90'].append(q90)
                df[var]['skew'].append(stats.skew(hist_list_joint[var],nan_policy='omit')*1.)
                df[var]['mean'].append(np.nanmean(hist_list_joint[var]))
                df[var]['std'].append(np.nanstd(hist_list_joint[var]))
                df[var]['cv'].append(np.nanstd(hist_list_joint[var])/np.nanmean(hist_list_joint[var]))
                df[var]['qcd'].append((q75-q25)/(q75+q25))
                df[var]['std_2'].append(np.nanstd(hist_list_joint[var][((hist_list_joint[var]>np.nanpercentile(hist_list_joint[var], 1)) & (hist_list_joint[var]<np.nanpercentile(hist_list_joint[var], 99)))]))

    pd.DataFrame(df).to_csv(metrics_dir+config['prefix']+'_metrics.csv')
                
            