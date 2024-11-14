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
plt.rcParams.update({'font.size': 14})


def iwc_num_integ(lam, mu, M0, a_mass_size, b_mass_size):
    Dlist = np.linspace(0.000001,0.05,100000)
    dD = Dlist[1]-Dlist[0]
    # df_subsel['lambda'] = ((df_subsel['mu']+3)/df_subsel['Deff'])
    # df_subsel['M0'] = 10**df_subsel['logM0']
    # df_subsel['a_mass_size'] = 10**df_subsel['loga_mass_size']

    psd_list = Dlist[None,:]**mu[:,None]*np.exp(-lam[:,None]*Dlist[None,:])

    N0_list = M0/ np.sum(psd_list*dD)
    iwc = np.sum((a_mass_size*N0_list)[:,None]*Dlist[None,:]**((mu+b_mass_size)[:,None])*np.exp(-lam[:,None]*Dlist[None,:])*dD,axis=1)

    return iwc




def compute_hist(path_root, targetspec, vars_to_hist=None, verbose=1, mse_threshold=100, after_burnin=0):

    # for i, path_root in enumerate(path_root_list):
    config_file = os.path.join(path_root,'config.json')
    path_data = path_root+'targetspec%02d/'%targetspec
    config = json.load(open(config_file,'r'))

    # Parameters in the theta vector (i.e., parameters to be inferred) and measurement keys (i.e., measurements to be used in the loss function)
    order_params = config['order_params'] # numbering of the parameters in the theta vector


    # Load the ground truth
    df_orig_data = pd.read_feather(config['data_params'])   
    ground_truth = df_orig_data.iloc[targetspec]
    ground_truth['logM0'] = np.log10(ground_truth['M0'])
    ground_truth['logDeff'] = np.log10(ground_truth['Deff'])
    lam_gt = ((ground_truth['mu']+3)/ground_truth['Deff'])
    M0_gt = ground_truth['M0']
    N0_gt =  M0_gt*lam_gt**(ground_truth['mu']+1)/special.gamma(ground_truth['mu']+1)
    ground_truth['iwc'] = special.gamma(ground_truth['mu']+ground_truth['b_mass_size']+1)*ground_truth['a_mass_size']/lam_gt**(ground_truth['mu']+ground_truth['b_mass_size']+1)*N0_gt
    ground_truth['loga_mass_size'] = np.log10(ground_truth['a_mass_size'])
    ground_truth['logalpha_area_size'] = np.log10(ground_truth['alpha_area_size'])

    if vars_to_hist is None:
        vars_to_hist = order_params

    # Load the chains     
    list_for_hist2 = {key:[] for key in vars_to_hist.keys()}
    list_for_hist2['iwc']=[]

    level = 0

    for chain_dir in sorted(glob.glob(path_data)):
        if verbose:
            print(chain_dir)
        for n_chain in range(100):
            if not os.path.exists(chain_dir+'theta_list_%03d.npy'%n_chain):
                continue

            theta_list= np.load(chain_dir+'theta_list_%03d.npy'%n_chain)
            mse_list = np.load(chain_dir+'mse_list_%03d.npy'%n_chain)[:,0]
            
            n = len(mse_list)
            theta_list = theta_list[:n, :,level]
            if (n<5002):
                if verbose:
                    print('chain too short')
                continue

            n_accepted = np.sum(mse_list[1:]-mse_list[:-1]!=0)
            inds = mse_list<mse_threshold
            theta_list = theta_list[inds]
            if verbose:
                print('perc_accepted',n_accepted/len(mse_list)*100)

            if (n_accepted/len(mse_list)*100<5):
                if verbose:
                    print('discarding chain with low acceptance rate: '+chain_dir+', chain #%03d'%n_chain)
                continue

            # A few calculations to get the IWC from the chains
            Deff=10**(theta_list[:,order_params['logDeff']])
            mu = theta_list[:,order_params['mu']]
            lam = ((mu+3)/Deff)
            b_mass_size = theta_list[:,order_params['b_mass_size']]
            a_mass_size = 10**theta_list[:,order_params['loga_mass_size']]
            M0 = 10**theta_list[:,order_params['logM0']]
            N0 = M0*lam**(mu+1)/special.gamma(mu+1)
            iwc = special.gamma(mu+b_mass_size+1)*a_mass_size/lam**(mu+b_mass_size+1)*N0

            inds_invalid = mu<=-1
            iwc[inds_invalid] = iwc_num_integ(lam[inds_invalid], mu[inds_invalid], M0[inds_invalid], a_mass_size[inds_invalid], b_mass_size[inds_invalid])
            
            list_for_hist2['iwc'].extend(iwc[after_burnin:].tolist())

            for key in vars_to_hist.keys():
                if key in order_params.keys():
                    list_for_hist2[key].extend(theta_list[after_burnin:,order_params[key]].tolist())

    return pd.DataFrame(list_for_hist2), ground_truth