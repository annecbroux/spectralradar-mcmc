from utils import *
import pandas as pd
import numpy as np
import os
import json 
import argparse
import shutil 
import copy

#----------------------------------------------------
## PARSE ARGS (CONFIG FILE)
#----------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_config', type=str, default='config.json')
args = parser.parse_args()

config_file = args.path_to_config

config = json.load(open(config_file,'r'))

if not os.path.exists(config['outdir_root']):
    os.makedirs(config['outdir_root'])

if os.path.exists(config['outdir_root']+'/config.json'):
    replace = input('config file already exists in %s, do you want to replace it? (y/n)'%config['outdir_root'])
    if replace=='n':
        exit()
shutil.copy2(config_file, config['outdir_root']+'/config.json')


#----------------------------------------------------
## LOAD DATA AND CONFIGURE MCMC CHAIN
#----------------------------------------------------

# Load all the data (and parameters - ground truth) of the various spectra / sets of moments on which we want to run the MCMC chain
params_from_dataset_total = pd.read_feather(config['data_params'])   
data_total = pd.read_feather(config['data_measurements'])

# Parameters in the theta vector (i.e., parameters to be inferred) and measurement keys (i.e., measurements to be used in the loss function)
order_params = config['order_params'] # numbering of the parameters in the theta vector
measurement_keys= config['measurement_keys']
# Load the original dataset (from which we will sample the initial theta and the proposal distribution) - NB this is not the same as the dataset on which we run the MCMC chain
# Actually (cf below) the data is then replaced with a uniform prior within the bounds of the original dataset
df_psd_orig = pd.read_feather(config['original_dataset'])   

# if wind 
dw_std = 0.5
if 'dw_std' in config.keys():
    dw_std = config['dw_std']

w_std = 1
if 'w_std' in config.keys():
    w_std = config['w_std']


# Complete the data with a uniform prior within the bounds of the original dataset
for param in order_params.keys():
    if param == 'logedr':
        df_psd_orig[param] = np.random.uniform(-5,-2.3, len(df_psd_orig))
    elif param == 'wind_w':
        df_psd_orig[param] = np.random.normal(0,w_std, len(df_psd_orig))
    elif param == 'dw_X':
        df_psd_orig[param] = np.random.normal(0,dw_std, len(df_psd_orig))
    elif param == 'dw_Ka':
        df_psd_orig[param] = np.random.normal(0,dw_std, len(df_psd_orig))
    elif param[:3]=='log':
        df_psd_orig[param] = np.log10(df_psd_orig[param[3:]])
        params_from_dataset_total[param] = np.log10(params_from_dataset_total[param[3:]])
    

#----------------------------------------------------
# Configure MCMC chain (Metropolis - Hastings / AP algorithm)
n_acc = config['n_acc']
n_steps = config['n_steps']
burnin = config['burnin']
sigma_meas = config['sigma_meas']
h_memory = config['h_memory']

coef_sigma_init = 1
if 'coef_sigma_init' in config.keys():
    coef_sigma_init = config['coef_sigma_init']

cov = 0 # whether or not to use covariance
if 'cov' in config.keys():
    cov = config['cov']

discard_noise = 0
if 'discard_noise' in config.keys():
    discard_noise = config['discard_noise']

n_acc_for_ap = 10 # Some parameters for adaptive proposal
if 'n_acc_for_ap' in config.keys():
    n_acc_for_ap = config['n_acc_for_ap']

ap_cov = 0 # Some parameters for adaptive proposal
if 'ap_cov' in config.keys():
    ap_cov = config['ap_cov']

swap = 2 # Swap frequency (for parallel tempering)
if 'swap' in config.keys():
    swap = config['swap']

update = 1 # Update frequency
if 'update' in config.keys():
    update = config['update']

ssrg_coefs = [0.17,0.23,1.67, 1.0] # Default SSRG coefficients (Hogan 2014)
if 'ssrg_coefs' in config.keys():
    ssrg_coefs = config['ssrg_coefs']

#----------------------------------------------------
# Parallel tempering
n_pt = 1
if 'n_pt' in config.keys():
    n_pt = config['n_pt']
    
if n_pt ==1:
    T = [1.]
elif n_pt == 5:
    T = [1., 2., 3., 4., 5.]


#----------------------------------------------------
# Some hyperparameters related to monitoring the chain
plot=config['plot']
print_res=config['print_res']
mse_thres=config['mse_thres']

#----------------------------------------------------
## DEFINE THE MAIN FUNCTION TO RUN THE MCMC CHAIN
#----------------------------------------------------

def run_mcmc(i_spec, figname=None,plot=1,outdir=None,print_res=1):
    """ 
    Run one instance of MCMC chain
    """
    params_from_dataset = params_from_dataset_total.iloc[i_spec]
    data = data_total.iloc[i_spec]

    if discard_noise: # in the measurements, set all the noise values to the noise level (this reduces the impact of the noisy part of the spectrum in the loss function)
        if 'spectra_X' in measurement_keys:
            data['spectra_X'] = np.array(data['spectra_X'])
            inds_X = data['spectra_X'] < 10*np.log10(params_from_dataset['noise_level_Ka'])+1
            data['spectra_X'][inds_X] = 10*np.log10(params_from_dataset['noise_level_Ka'])
        
        if 'spectra_Ka' in measurement_keys:
            data['spectra_Ka'] = np.array(data['spectra_Ka'])
            inds_Ka = data['spectra_Ka'] < 10*np.log10(params_from_dataset['noise_level_Ka'])+1
            data['spectra_Ka'][inds_Ka] = 10*np.log10(params_from_dataset['noise_level_Ka'])

        if 'spectra_W' in measurement_keys:
            data['spectra_W'] = np.array(data['spectra_W'])
            inds_W = data['spectra_W'] < 10*np.log10(params_from_dataset['noise_level_Ka'])+1
            data['spectra_W'][inds_W] = 10*np.log10(params_from_dataset['noise_level_Ka'])

    # initialize variables and containers
    step = 0
    n_accepted = [0 for i in range(n_pt)]
    n_swap = np.zeros(n_pt)
    theta_list = []
    log_likelihood_list = []
    mse_list = []
    mse_accepted_list = [[] for i in range(n_pt)]
    theta_accepted_list = [[] for i in range(n_pt)]
    chain_status_list = []

    # 1) initialize theta from prior
    theta = initialize_theta_from_prior_pt(df_psd_orig,order_params, n_pt)

    if cov:
        df_mass_loga_fit = pd.read_csv('~/mcmc_data/parameters/mass_size_log_loga',sep='\t')
        df_area_logalpha_fit = pd.read_csv('~/mcmc_data/parameters/area_size_log_loga',sep='\t')

        for i in range(n_pt):
            theta[order_params['loga_mass_size'],i] = np.log10(sample_from_fit(df_mass_loga_fit[df_mass_loga_fit['ptype']=='agg'],[theta[order_params['b_mass_size'],i]])[0])
            theta[order_params['logalpha_area_size'],i] = np.log10(sample_from_fit(df_area_logalpha_fit[df_area_logalpha_fit['ptype']=='agg'],[theta[order_params['beta_area_size'],i]])[0])    

            if 'loga_mass_size_2' in order_params.keys():
                theta[order_params['loga_mass_size_2'],i] = np.log10(sample_from_fit(df_mass_loga_fit[df_mass_loga_fit['ptype']=='agg'],[theta[order_params['b_mass_size_2'],i]])[0])
                theta[order_params['logalpha_area_size_2'],i] = np.log10(sample_from_fit(df_area_logalpha_fit[df_area_logalpha_fit['ptype']=='agg'],[theta[order_params['beta_area_size_2'],i]])[0])


    # turb can be free or not
    if 'logedr' in order_params.keys():
        edr = 10**theta[order_params['logedr']]
        print('Turbulent EDR is a free parameter')
    else:
        edr = config['turb_edr']*np.ones(n_pt)
        print('Turbulent EDR is fixed to %.2f'%edr[0])

    # freeze some parameters to a fixed value or to their true value
    for key in config['freeze_params']:
        theta[order_params[key]] = config['freeze_params'][key]
    for key in config['freeze_params_to_true']:
        theta[order_params[key]] = params_from_dataset[key]


    # 1bis) forward model with initial theta and compute loss metrics
    pred = [forward_model(theta[:,i], params_from_dataset, order_params,edr=edr[i], ssrg_coefs=ssrg_coefs) for i in range(n_pt)]
    l = [compute_log_likelihood(data,pred[i], sigma_meas=sigma_meas,data_keys=measurement_keys) for i in range(n_pt)]
    mse = [compute_mse(data,pred[i], data_keys=measurement_keys)[0] for i in range(n_pt)]
    p = [compute_prior_log_prob(theta[:,i],df_psd_orig,order_params) for i in range(n_pt)]
    chain_status = [i for i in range(n_pt)]

    theta_list.append(theta)
    log_likelihood_list.append(l)
    mse_list.append(mse)
    chain_status_list.append(chain_status)

    # 1ter) initialize proposal distribution
    # sigma_init is the covariance to use initially for the proposal 
    if cov:
        sigma_init = [initial_sigma_proposal_cov(df_psd_orig,order_params)*coef_sigma_init*7/len(order_params) for i in range(n_pt)] # formula from Tamminen 2001
    else:
        sigma_init = [initial_sigma_proposal(df_psd_orig,order_params)*coef_sigma_init for i in range(n_pt)]
    cov_init = [initial_sigma_proposal_cov(df_psd_orig,order_params) for i in range(n_pt)]

    # Loop MCMC steps
    while ((n_accepted[0] < n_acc) and (step < n_steps)): # stopping conditions

        # 2) sample theta_star from proposal distribution ######################################
        
        if step < 100:
            sigma_proposal = copy.deepcopy(sigma_init)
        
        for i in range(n_pt):  # parallel tempering: go through mcmc step for all chains

            # Very first steps: need to "manually" adjust the proposal variance
            if (step > 100) & (n_accepted[i] < 6):
                sigma_proposal[i] = copy.deepcopy(sigma_init[i])/5
            elif (step > 800) & (n_accepted[i] < 20):
                sigma_proposal[i] = copy.deepcopy(sigma_init[i])/10

            if ((step > 0) & (step%h_memory==0) & (n_accepted[i] > n_acc_for_ap) & (n_accepted[i] < burnin)): # burnin steps
                sigma_proposal[i] = np.cov(np.array(theta_accepted_list[i])[-h_memory:].T)*2.38**2/len(order_params)
                print('updating sigma_proposal for AP Rosenthal')
                
        theta_star = np.zeros((len(order_params),n_pt))
        for i in range(n_pt):
            theta_star[:,i] = sample_from_proposal(theta[:,i],sigma_proposal[i])
            

        # turb can be free or not
        if 'logedr' in order_params.keys():
            edr = 10**theta_star[order_params['logedr']]
        else:
            edr = config['turb_edr']*np.ones(n_pt)

        # freeze some parameters to a fixed value or to their true value
        for key in config['freeze_params']:
            theta_star[order_params[key]] = config['freeze_params'][key]
        for key in config['freeze_params_to_true']:
            theta_star[order_params[key]] = params_from_dataset[key]

            
        # 3) forward model with theta_star and compute loss metrics ##############################################
        pred_star = [forward_model(theta_star[:,i], params_from_dataset, order_params, edr=edr[i], discard_noise = discard_noise, ssrg_coefs=ssrg_coefs) for i in range(n_pt)]
        l_star = [compute_log_likelihood(data,pred_star[i],sigma_meas=sigma_meas,data_keys=measurement_keys) for i in range(n_pt)]
        mse_star = [compute_mse(data,pred_star[i], data_keys=measurement_keys)[0] for i in range(n_pt)]
        p_star = [compute_prior_log_prob(theta_star[:,i],df_psd_orig,order_params) for i in range(n_pt)] 
        
        loss = np.zeros(n_pt)
        for i in range(n_pt):
            loss[i] = 1/T[i]*(l_star[i] - l[i] + p_star[i] - p[i]) # note that prior can also be out of the division by T

        loss_swap = np.zeros((n_pt, n_pt))
        for i in range(n_pt):
            for j in range(n_pt):
                loss_swap[i,j] = (l[j]+p[j]-l[i]-p[i])/T[i]+(l[i]+p[i]-l[j]-p[j])/T[j] # note that prior can also be out of the division by T

        if (step%10==0) & print_res:
            print(step,loss,mse,mse_star, l, l_star)


        # 4) accept or reject theta_star: normal update ###############################################################

        if step%update==0:
            u = np.log(np.random.uniform(0,1, n_pt))
            for i in range(n_pt):
                if (loss[i] > u[i]):
                    if i==0:
                        print('accept = %d, step = %d, loss = %.1f, mse_old = %.1f, mse_new = %.1f'%(n_accepted[0],step,loss[0],mse[0],mse_star[0]))
                    theta[:,i] = theta_star[:,i]*1.
                    pred[i] = pred_star[i]
                    l[i] = l_star[i]*1.
                    mse[i] = mse_star[i]*1.
                    p[i] = p_star[i]*1.
                    n_accepted[i] += 1
                    mse_accepted_list[i].append(mse[i])
                    theta_accepted_list[i].append(copy.deepcopy(theta[:,i]))
        

        # 4bis) parallel tempering: chain status update ###############################################################

        chain_status = np.arange(n_pt) # initialize chain status for the current step
        u = np.log(np.random.uniform(0,1, (n_pt, n_pt)))
        for i in range(n_pt):
            for j in range(i+1,n_pt):
                if (step%swap==0) & (loss_swap[i,j] > u[i,j]):
                    theta[:,i], theta[:,j] = theta[:,j]*1., theta[:,i]*1.
                    pred[i], pred[j] = pred[j], pred[i]
                    l[i], l[j] = l[j]*1., l[i]*1.
                    mse[i], mse[j] = mse[j]*1., mse[i]*1.
                    p[i], p[j] = p[j]*1., p[i]*1.
                    n_swap[i] += 1
                    n_swap[j] += 1
                    chain_status[i]=j #chain_status_list[-1][j] 
                    chain_status[j]=i #chain_status_list[-1][i]


        # 5) print results and plot if necessary ########################################################
        if step%500==0 and (plot):
            fig = plt.figure()
            title = str(mse[0])
            if 'spectra_X' in measurement_keys:
                plt.plot(data['spectra_X'], 'k',label='data_X')
                plt.plot(pred[0]['spectra_X'][0],'--k',label='pred_X')
                plt.plot(pred_star[0]['spectra_X'][0],':k',label='pred_star_X')
                title = title+' -- '+str(np.mean((data['spectra_X']-pred[0]['spectra_X'][0])**2))
            if 'spectra_Ka' in measurement_keys:
                plt.plot(data['spectra_Ka'], 'b',label='data_Ka')
                plt.plot(pred[0]['spectra_Ka'][0], '--b',label='pred_Ka')
                plt.plot(pred_star[0]['spectra_Ka'][0], ':b',label='pred_star_Ka')
                title = title +' -- '+str(np.mean((data['spectra_Ka']-pred[0]['spectra_Ka'][0])**2))
            if 'spectra_W' in measurement_keys:
                plt.plot(data['spectra_W'], 'r',label='data_W')
                plt.plot(pred[0]['spectra_W'][0], '--r', label='pred_W')
                plt.plot(pred_star[0]['spectra_W'][0], ':r', label='pred_star_W')
                title = title+' -- '+str(np.mean((data['spectra_W']-pred[0]['spectra_W'][0])**2))
            plt.title(title)
            plt.legend()
            fig.savefig(figname+'_%d.png'%step)
            plt.close()

        theta_list.append(np.array(theta))
        log_likelihood_list.append(np.array(l))
        mse_list.append(np.array(mse))
        chain_status_list.append(np.array(chain_status))
        step += 1

        # Some conditions for early stop of chain, if getting stuck
        n_accepted_for_kill = np.sum(np.array(mse_list)[1:,0]-np.array(mse_list)[:-1,0]!=0)
        if len(mse_list)>500:
            n_accepted_latest = np.sum(np.array(mse_list)[-500:,0]-np.array(mse_list)[-501:-1,0]!=0)
        else:
            n_accepted_latest = n_accepted_for_kill
        if (step > 5000) and (np.all(np.array(mse_accepted_list[0])[-50:]>=15)):
            print('stuck in bad local minimum, exiting')
            break
        if (step > 5000) and (n_accepted_for_kill/step < 0.01):
            print('chain not moving, exiting')
            break
        if (step>=10000) & (n_accepted_for_kill/step*100<8) & (n_accepted_latest/500*100<10):
            break
        if (step>=20000) & (n_accepted_for_kill/step*100<10):
            break
    
    return theta_list, log_likelihood_list, mse_list, pred, data, chain_status_list, n_accepted, n_swap
    

if __name__ == '__main__':

    for chain_number in range(config['n_chains']):
        for i_spec in range(config['n_spec_start'], config['n_spec_end']):
            print('\n---------------------\nCHAIN # %d -- SPEC # %d\n---------------------\n'%(chain_number,i_spec))

            outdir = config['outdir_root']+'/targetspec%02d'%i_spec
            if os.path.exists(outdir+'/theta_list_%03d.npy'%chain_number):
                print('chain already run, skipping')
                continue
            
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            
            theta_list, log_likelihood_list, mse_list, pred, data, chain_status, n_accepted_list, n_swap = run_mcmc(i_spec, figname=outdir+'/pred_%03d'%chain_number, outdir=outdir, plot=plot, print_res=print_res)

            np.save(outdir+'/theta_list_%03d.npy'%chain_number,np.array(theta_list))
            np.save(outdir+'/mse_list_%03d.npy'%chain_number,np.array(mse_list))
            np.save(outdir+'/chain_status_%03d.npy'%chain_number,np.array(chain_status))
            np.save(outdir+'/n_acc_%03d.npy'%chain_number,np.array(n_accepted_list))
            np.save(outdir+'/n_swap_%03d.npy'%chain_number,np.array(n_swap))
