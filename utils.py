import pyPamtra
import numpy as np
import matplotlib.pyplot as plt
import sys
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
import scipy
import random
import pandas as pd
from scipy.interpolate import UnivariateSpline
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def kde(x, bandwidth=0.2, **kwargs):
    """
    Kernel Density Estimation with Scipy
    x: array-like input data 
    bandwidth: float, bandwidth parameter for the Gaussian kernels.

    returns: gaussian_kde instance with Scipy API
    """

    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde


def generate_rand_from_pdf(pdf, x_grid, n_points=100):
    """
    Generate random numbers from a given PDF (discretized)
    pdf: discrete probability density function / histogram
    x_grid: values of the random variable corresponding to the pdf values / histogram bins
    n_points: number of random values to generate

    returns: random values drawn from the given pdf
    """

    cdf = np.cumsum(pdf)
    cdf = cdf / cdf[-1]
    values = np.random.rand(n_points)
    value_bins = np.searchsorted(cdf, values)
    random_from_cdf = x_grid[value_bins]
    return random_from_cdf



def initialize_theta_from_prior(df, order_params):
    """
    Initialize the state parameters from the prior distribution
    df: pandas dataframe containing the original dataset (used as prior)
    order_params: dictionary containing the order of the parameters in the theta vector
    
    returns: theta vector containing the initialized parameters
    """

    theta = np.zeros(len(order_params))-999
    for k in order_params.keys():
        data = df[k].to_numpy()
        x_grid = np.linspace(min(data), max(data), 1000)
        kdepdf = kde(data, bandwidth=np.mean(x_grid)/20).evaluate(x_grid)
        random_from_kde = generate_rand_from_pdf(kdepdf, x_grid, n_points=1)
        theta[order_params[k]] = random_from_kde[0]
    return theta


def initialize_theta_from_prior_pt(df, order_params, n_pt):
    """
    Initialize the state parameters from the prior distribution
    df: pandas dataframe containing the original dataset (used as prior)
    order_params: dictionary containing the order of the parameters in the theta vector
    
    returns: theta vector containing the initialized parameters
    """

    theta = np.zeros((len(order_params), n_pt))-999
    for i in range(n_pt):
        for k in order_params.keys():
            data = df[k].to_numpy()
            x_grid = np.linspace(min(data), max(data), 1000)
            kdepdf = kde(data, bandwidth=np.mean(x_grid)/20).evaluate(x_grid)
            random_from_kde = generate_rand_from_pdf(kdepdf, x_grid, n_points=1)
            theta[order_params[k],i] = random_from_kde[0]
    return theta


def sample_from_fit(fit_params,b):
    loga = (b+np.random.normal(0,5*np.array(fit_params['rmse']),len(b))-np.array(fit_params['intercept']))/np.array(fit_params['slope'])
    a = np.exp(loga)
    return a


def compute_prior_log_prob(theta, df, order_params):
    """
    Compute the log probability of theta according to the prior distribution
    theta: vector containing the state parameters
    df: pandas dataframe containing the original dataset (used as prior)
    order_params: dictionary containing the order of the parameters in the theta vector
    
    returns: log probability of theta
    """

    log_prob = 0
    for k in order_params.keys():
        # if k == 'wind_w':
        #     log_prob += np.log(stats.norm(0,1).pdf(theta[order_params[k]]))
        # else:
        data = df[k].to_numpy()
        hist, bins = np.histogram(data, bins=100)
        x_grid = np.linspace(min(data), max(data), 1000)
        p_from_kde = kde(data, bandwidth=np.mean(x_grid)/20).evaluate(theta[order_params[k]])
        log_prob += np.log(p_from_kde)
    return log_prob


def initial_sigma_proposal(df, order_params):
    """
    Compute the initial sigma of the proposal distribution for the MCMC. This assumes diagonal covariance.
    df: pandas dataframe containing the original dataset (used as prior)
    order_params: dictionary containing the order of the parameters in the theta vector

    returns: vector containing the initial sigma of the proposal distribution   
    """

    sigma_proposal = np.zeros(len(order_params))
    for k in order_params.keys():
        data = df[k].to_numpy()
        sigma_proposal[order_params[k]] = np.std(data)
    return sigma_proposal


def initial_sigma_proposal_cov(df, order_params):
    """
    Compute the initial covariance of the proposal distribution for the MCMC
    df: pandas dataframe containing the original dataset (used as prior)
    order_params: dictionary containing the order of the parameters in the theta vector
    """
    df_to_np = df[order_params.keys()].to_numpy()
    cov = np.cov(df_to_np.T)
    return cov


def sample_from_proposal(theta,sigma):
    """
    Sample a new state vector from the proposal distribution (Gaussian)
    theta: reference state vector
    sigma: vector containing the sigma of the proposal distribution (here we assume independent parameters - diagonal covariance matrix represented with 1D vector)

    returns: theta_star new state vector sampled from the proposal distribution
    """
    if sigma.ndim==1:
        theta_star = np.zeros_like(theta)
        for i in range(len(theta)):
            theta_star[i] = np.random.normal(loc=theta[i],scale=sigma[i])
    else:
        theta_star = np.random.multivariate_normal(mean=theta,cov=sigma)
    return theta_star


def sample_from_proposal_AP(theta,Ktilde,scale=2.4):
    """
    Sample a new state vector from the proposal distribution (Gaussian) following the AP version of Metropolis-Hastings
    """
    h = Ktilde.shape[1]
    g = np.random.normal(0,1,(h,1))
    s = scale/np.sqrt(Ktilde.shape[0])/np.sqrt(h-1)
    theta_star = theta + s*np.matmul(Ktilde,g)[:,0]
    return theta_star


def compute_log_likelihood(data,pred,data_keys = ['spectra_X','spectra_Ka','spectra_W'],sigma_meas=1.):
    """
    Compute the log likelihood of the measurement data given the prediction
    data: dictionary containing the measurement data (spectra from radar)
    pred: dictionary containing the prediction (spectra from forward model)
    data_keys: list of keys to consider in the data dictionary (defaults to spectra at 3 freqs, could also be moments, Ze,...)
    sigma_meas: standard deviation of the measurement noise (assumed to be Gaussian, unbiased)

    returns: log likelihood of the measurement data given the prediction (Gaussian likelihood model)
    """

    log_likelihood_dict = {}
    for key in data_keys:
        if hasattr(sigma_meas, '__len__'):
            sigma_meas_err = sigma_meas[key]
        else:
            sigma_meas_err = sigma_meas
        
        if data[key].ndim>1:
            log_likelihood_dict[key] = 0
            for i in range(len(data[key])):
                log_likelihood_dict[key] += -0.5*np.sum((np.array(data[key][i])-np.array(pred[key][i]))**2)/(2*sigma_meas_err**2) - 0.5*np.log(2*np.pi*sigma_meas_err**2)*len(data[key][i]) #not sure if I should divide by the number of points in the spectra - doesn't seem to change much
        else:
            try:
                log_likelihood_dict[key] = -0.5*np.sum((np.array(data[key])-np.array(pred[key]))**2)/(2*sigma_meas_err**2) - 0.5*np.log(2*np.pi*sigma_meas_err**2)*len(data[key])
            except:
                log_likelihood_dict[key] = -0.5*np.sum((np.array(data[key])-np.array(pred[key]))**2)/(2*sigma_meas_err**2) - 0.5*np.log(2*np.pi*sigma_meas_err**2)

        if 'spectra' in key:
                log_likelihood_dict[key] /= 256

        log_likelihood = np.sum(list(log_likelihood_dict.values()))/len(data_keys)
    return log_likelihood




def compute_mse(data,pred,data_keys = ['spectra_X','spectra_Ka','spectra_W']):
    """
    Compute the mean squared error between the measurement data and the prediction
    data: dictionary containing the measurement data (spectra from radar)
    pred: dictionary containing the prediction (spectra from forward model)
    data_keys: list of keys to consider in the data dictionary (defaults to spectra at 3 freqs, could also be moments, Ze,...)

    returns: mean squared error between the measurement data and the prediction
    """

    # mse = 0
    mse_dict = {}
    for key in data_keys:
        if data[key].ndim>1:
            mse_dict[key] = np.mean((np.array(data[key][0])-np.array(pred[key][0]))**2)
        else:
            mse_dict[key] = np.mean((np.array(data[key])-np.array(pred[key]))**2)
    mse = np.sum(list(mse_dict.values()))/len(data_keys)
    return mse, mse_dict




def forward_model(theta, params_from_dataset, order_params, force_exp=0, edr=5e-3, block_beta=0,discard_noise=0, ssrg_coefs=[0.19,0.23,1.67, 1.0]):
    """ 
    Compute the forward model prediction given the state vector
    theta: state vector containing the state parameters
    params_from_dataset: dictionary containing the parameters from the dataset (e.g. temperature, pressure, etc.) which are additionally needed
    order_params: dictionary containing the order of the parameters in the theta vector

    returns: dictionary containing the forward model prediction (spectra at 3 frequencies + moments + Ze)
    """

    ########### DEFINE ENVIRONMENTAL PARAMETERS ###################
    rho_air = 1.2

    ########### DEFINE MICROPHYSICAL PARAMETERS ##################

    if 'M0' in order_params.keys():
        Nt = theta[order_params['M0']] 
    elif 'logM0' in order_params.keys():
        Nt = 10**theta[order_params['logM0']]
    else:
        raise ValueError('M0 or its log not in order_params')
    if 'Deff' in order_params.keys():
        Deff = theta[order_params['Deff']]
    elif 'logDeff' in order_params.keys():
        Deff = 10**theta[order_params['logDeff']]
    else:
        raise ValueError('Deff or its log not in order_params')

    use_radar_airmotion = False
    if "wind_w" in order_params.keys():
        use_radar_airmotion = True

    aspect_ratio = 0.6

    canting_angle = 0.

    ######### RESHAPE PARAMETERS CORRECTLY ##################      

    pam = pyPamtra.pyPamtra()
    if force_exp:
        mu = 0
    elif 'mu' not in order_params.keys():
        mu = theta[order_params['mu_uniform']]
    else:
        mu = theta[order_params['mu']]

    if 'a_mass_size' in order_params.keys():
        ams = theta[order_params['a_mass_size']]
    elif 'loga_mass_size' in order_params.keys():
        ams = 10**theta[order_params['loga_mass_size']]
    else:
        raise ValueError('a_mass_size or its log not found in order_params')

    if 'alpha_area_size' in order_params.keys():
        aas = theta[order_params['alpha_area_size']]
    elif 'logalpha_area_size' in order_params.keys():
        aas = 10**theta[order_params['logalpha_area_size']]
    else:
        raise ValueError('alpha_area_size or its log not found in order_params')

    pam.df.addHydrometeor(('ice0', #name
                            aspect_ratio, #aspect ratio, <1 means oblate
                            -1, #phase: -1=ice, 1=liq
                            -99.,#200, #density
                            ams, #a parameter of mass-size
                            theta[order_params['b_mass_size']], #b parameter of mass-size
                            aas, #alpha parameter of cross-section area - size relation
                            theta[order_params['beta_area_size']], #beta parameter of cross-section area - size relation
                            12, # moment provided in input file
                            200, #number of discrete size bins (internally, nbins+1 is used)
                            'mgamma', #name of psd
                            -99., #1st parameter of psd
                            -99.,#2nd parameter of psd
                            mu, #3rd parameter of psd
                            1., #4th parameter of psd
                            1e-5, #min diameter
                            1e-2, # max diameter
                            # 'ss-rayleigh-gans',
                            'ss-rayleigh-gans_%.3f_%.3f_%.3f_%.3f'%tuple(ssrg_coefs),
                            'heymsfield10_particles', 
                            canting_angle)) #canting angle of hydrometeors, only for Tmatrix and SSRG
    
    if ('Deff_2' in order_params.keys()) | ('logDeff_2' in order_params.keys()):
        if 'a_mass_size_2' in order_params.keys():
            ams_2 = theta[order_params['a_mass_size_2']]
        elif 'loga_mass_size_2' in order_params.keys():
            ams_2 = 10**theta[order_params['loga_mass_size_2']]
        else:
            raise ValueError('a_mass_size_2 or its log not found in order_params')

        if 'alpha_area_size_2' in order_params.keys():
            aas_2 = theta[order_params['alpha_area_size_2']]
        elif 'logalpha_area_size_2' in order_params.keys():
            aas_2 = 10**theta[order_params['logalpha_area_size_2']]
        else:
            raise ValueError('alpha_area_size or its log not found in order_params')
        
        if 'M0_2' in order_params.keys():
            Nt2 = theta[order_params['M0_2']]
        elif 'logM0_2' in order_params.keys():
            Nt2 = 10**theta[order_params['logM0_2']]

        pam.df.addHydrometeor(('ice1', #name
                            aspect_ratio, #aspect ratio, <1 means oblate
                            -1, #phase: -1=ice, 1=liq
                            -99.,#200, #density
                            ams_2, #a parameter of mass-size
                            theta[order_params['b_mass_size_2']], #b parameter of mass-size
                            aas_2, #alpha parameter of cross-section area - size relation
                            theta[order_params['beta_area_size_2']], #beta parameter of cross-section area - size relation
                            12, # moment provided in input file
                            200, #number of discrete size bins (internally, nbins+1 is used)
                            'mgamma', #name of psd
                            -99., #1st parameter of psd
                            -99.,#2nd parameter of psd
                            theta[order_params['mu_2']], #3rd parameter of psd
                            1., #4th parameter of psd
                            1e-5, #min diameter
                            1e-2, # max diameter
                            'ss-rayleigh-gans',
                            'heymsfield10_particles', 
                            canting_angle)) #canting angle of hydrometeors, only for Tmatrix and SSRG

    pam = pyPamtra.importer.createUsStandardProfile(pam,hgt_lev=np.array([np.linspace(999,1001,2).tolist()]))
    pam.p['temp_lev'][:]=params_from_dataset['temperature']+273.15
    pam.p['relhum_lev'][:]=90.
    pam.set["verbose"] = -1

    pam.p["hydro_n"][0,0,:,0] = Nt/rho_air
    pam.p["hydro_reff"][0,0,:,0] = 1/2*Deff
    
    if 'Deff_2' in order_params.keys():
        pam.p["hydro_n"][0,0,:,1] = Nt2/rho_air
        pam.p["hydro_reff"][0,0,:,1] = 1/2*theta[order_params['Deff_2']]
    if 'logDeff_2' in order_params.keys():
        pam.p["hydro_n"][0,0,:,1] = Nt2/rho_air
        pam.p["hydro_reff"][0,0,:,1] = 1/2*10**theta[order_params['logDeff_2']]

    
    pam.nmlSet["radar_mode"] = "spectrum"
    pam.nmlSet['save_psd']=True
    pam.nmlSet["radar_noise_distance_factor"] = 1.0
    pam.nmlSet["radar_save_noise_corrected_spectra"]=  False
    pam.nmlSet['passive'] = False
    pam.nmlSet['radar_airmotion'] = use_radar_airmotion
    pam.nmlSet['radar_aliasing_nyquist_interv'] = 1
    pam.nmlSet['obs_height'] = 0
    pam.nmlSet['hydro_adaptive_grid'] = False
    pam.nmlSet['radar_allow_negative_dD_dU'] = True

    pam.nmlSet['radar_nfft']= int(params_from_dataset['fft_len_W'])
    pam.nmlSet['radar_max_v']= params_from_dataset['v_nyq_W']
    pam.nmlSet['radar_min_v']= -params_from_dataset['v_nyq_W']
    pam.addSpectralBroadening(edr, 10, params_from_dataset['beamwidth_deg_W'], params_from_dataset['integration_time_W'], params_from_dataset['freq_W'], kolmogorov=0.5)
    
    vel0 = np.linspace(-params_from_dataset['v_nyq_W'],params_from_dataset['v_nyq_W'],int(params_from_dataset['fft_len_W']),endpoint=False)
    dv0 = vel0[1]-vel0[0]
    pam.nmlSet['radar_pnoise0'] = 10*np.log10(np.array(params_from_dataset['noise_level_Ka']))+10*np.log10(dv0*params_from_dataset['fft_len_Ka'])


    pred = {}
    for freq in ['X','Ka','W']:
        if use_radar_airmotion:
           pam.p["wind_w"][:] = theta[order_params['wind_w']]
        else:
            pam.p["wind_w"][:] = params_from_dataset['wind_w_W']

        if ((freq != 'W') & (use_radar_airmotion) & ('dw_%s'%freq in order_params.keys())):
            pam.p['wind_w'][:] = pam.p['wind_w']+theta[order_params['dw_%s'%freq]]
        
        try:
            pam.runPamtra(params_from_dataset['freq_%s'%freq])
        except Exception as e:
            pam.r['radar_spectra']=np.zeros((1, 1, 1, 1, 1, 256))-9999.
            pam.r['radar_moments']=np.zeros((1, 1, 1, 1, 1, 1, 4))-9999.
            pam.r['Ze']=np.zeros((1, 1, 1, 1, 1, 1))-9999.


        if block_beta:
            if theta[order_params['beta_area_size']]<1:
                pam.r['radar_spectra']=pam.r['radar_spectra']*0.-9999.
                pam.r['radar_moments']=pam.r['radar_moments']*0.-9999.
                pam.r['Ze']=pam.r['Ze']*0.-9999.

        pred['spectra_%s'%freq]=pam.r['radar_spectra'][0,0,0,0,0,:]-10*np.log10(dv0)
        pred['moments_%s'%freq] = pam.r["radar_moments"][0,0,0,0,0,0,:].tolist() 
        pred['Ze_%s'%freq] = pam.r["Ze"][0,0,0,0,0,0]
        pred['mdv_%s'%freq] = pam.r["radar_moments"][0,0,0,0,0,0,0]
        pred['sw_%s'%freq] = pam.r["radar_moments"][0,0,0,0,0,0,1]
        pred['sk_%s'%freq] = pam.r["radar_moments"][0,0,0,0,0,0,2]


    if discard_noise: # we are not interested in the error that comes from the noisy part of the spectrum -> set it to the noise level
        inds_X = pred['spectra_X'] < 10*np.log10(params_from_dataset['noise_level_Ka'])+1
        pred['spectra_X'][inds_X] = 10*np.log10(params_from_dataset['noise_level_Ka'])
        inds_Ka = pred['spectra_Ka'] < 10*np.log10(params_from_dataset['noise_level_Ka'])+1
        pred['spectra_Ka'][inds_Ka] = 10*np.log10(params_from_dataset['noise_level_Ka'])
        inds_W = pred['spectra_W'] < 10*np.log10(params_from_dataset['noise_level_Ka'])+1
        pred['spectra_W'][inds_W] = 10*np.log10(params_from_dataset['noise_level_Ka'])
    
    pred['spectra_X'] = [pred['spectra_X']]
    pred['spectra_Ka'] = [pred['spectra_Ka']]
    pred['spectra_W'] = [pred['spectra_W']]

    return pred

