#################################################
## PHY327 Data Analysis Nonlinear Fitting Code ##
#################################################
## Version: 2.0 ## Author: Deni Straus ##########
#################################################
## This code is designed to fit unordered data ##
## with uncertainties to gaussian, lognormal,  ##
## and laplacian functions. It also calculates ##
## the reduced Chi-squared of each fit and     ##
## residuals.                                  ##
#################################################
## Update 2.0:                                 ##
##  - Added both ODR and least-sqaures fits    ##
##  - Moved fitting and plotting to seperate   ##
##    functions outside of main function       ##
##  - Added various helper functions           ##
#################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import matplotlib.font_manager as fm
import scipy.optimize as spop
import scipy.odr as odr

#Helper function that rounds uncertainties and values based on their uncertainty
def ez_round(x, dx):
    if dx//1 != 0:
        flag1 = 1
        flag2 = 1
        i1, i2, j, k = 1, 1, 0, 0
        while flag1 != 0.0:
            j+=1
            i1*=10
            flag1 = dx//i1
        while flag2 >= 1.0:
            k+=1
            i2*=10
            flag2 = x//i2
        x = int(round(x/i2, (k-j+1))*i2)
        dx = int(round(dx/i1, 1)*i1)
    else:
        flag3 = 0
        a = 0
        while flag3 == 0.0:
            a += 1
            flag3 = round(dx, a)
        x = round(x, a)
        dx = round(dx, a)

    return [x, dx]

#Normal/Gaussian function definition
def gauss(x, *p)->float:
    mu = p[0]
    sigma = p[1]
    y_mu = p[2]
    return y_mu * np.exp(-((x - mu)**2/(2*sigma**2)))

#Lognormal function definition
def lognormal(x, *d)->float:
    mu = d[0]
    sigma = d[1]
    y_mu = d[2]
    return ((y_mu)/(x*sigma))*np.exp(-((np.log(x)-mu)**2/(2*sigma**2)))

#Laplacian function definition
def laplacian(x, *q)->float:
    mu = q[0]
    sigma = q[1]
    y_mu = q[2]
    return y_mu * np.exp(-((np.abs(x - mu))/(sigma)))

#Reduced Chi-squared calculation
def red_chi_sq(x: float, y: float, f, dy: float, mu: float, std_dev: float, y_mu: float,)->float:
    return (np.sum(((y-f(x,y_mu,mu,std_dev))/(dy))**2))/(len(x)-3)

#Helper function that loads data files
def load_data(filename: str):
    data = np.loadtxt(filename, comments='#', unpack=True, delimiter=',')
    return data

#Propogated uncertainty calculations and value scaling
def calc_uncertainty(x, y, dx, dy, s, scale_f: float, dscale_f: float):
    #Original data values are saved to a temporary variable and then scaled using the scaling factors
    #as well as the calibration factor. Then, using these values, propogated uncertainty for x and y
    #values are calculated
    for i in range(0, len(x)):
        original_x = x[i]
        original_y = y[i]
        x[i] = x[i] * s[i] * scale_f
        y[i] = y[i] * s[i] * scale_f

        dx[i] = x[i] * np.sqrt((dx[i]/original_x)**2 + (dscale_f/scale_f)**2)
        dy[i] = y[i] * np.sqrt((dy[i]/original_y)**2 + (dscale_f/scale_f)**2)

        x[i] = int(x[i])
        y[i] = int(y[i])
        dx[i] = int(dx[i])
        dy[i] = int(dy[i])
    return x, y, dx, dy

#Orthogonal Distance Regression (ODR) is useful for fitting functions when a data set has error in 
#both the indpendent and dependent variable (x and y), like this data set
def odr_fit(x, y, fun, dx, dy):
    #Helper function (needed for odr function, not important)
    def swapped(a,b):
        return fun(b,*a)
    
    #Guess mu and y_mu as the average of the x and y data
    y_mu_guess = float((sum(y))/(len(y)))*2
    mu_guess = float((sum(x))/(len(x)))
    #A good estimation for standard deviation is the span of your range of x values
    #divided by 4
    sigma_guess = float((max(x)-min(x))/2)

    param_guess = [mu_guess, sigma_guess, y_mu_guess]
    
    #ODR data fit, outputs fit parameters with uncertainties, as well as the quasi-chi-squared
    data = odr.RealData(x=x, y=y, sx=dx, sy=dy)
        
    model = odr.Model(swapped)
    fitted = odr.ODR(data, model, beta0=param_guess, maxit=50, job=0)
    output = fitted.run()

    params = output.beta
    chisq = output.res_var
    params_err = output.sd_beta

    #Makes sure that uncertainties aren't scaled down
    if chisq < 1.0 :
        params_err = params_err/np.sqrt(chisq)

    #This segment is taken from the ODR_fitting code from https://www.physics.utoronto.ca/apl/python/index.htm#repository

    #estimated x and y component of the residuals
    delta   = output.delta  
    epsilon = output.eps

    #These are the projections of dx and dy on the residual line
    dx_star = ( dx*np.sqrt( ((dy*delta)**2) /
                ( (dy*delta)**2 + (dx*epsilon)**2 ) ) )
    dy_star = ( dy*np.sqrt( ((dx*epsilon)**2) /
                ( (dy*delta)**2 + (dx*epsilon)**2 ) ) )
    sigma_odr = np.sqrt(dx_star**2 + dy_star**2)
    # residual is positive if the point lies above the fitted curve, negative if below
    residual = ( np.sign(y-fun(x,*params))
                * np.sqrt(delta**2 + epsilon**2) )
    
    return params, chisq, params_err, sigma_odr, residual

#Non-linear least-sqaures uses the least squares method to fit data to non-linear functions. Only
#considers uncertainty in the dependent variable.
def nl_ls_fit(x, y, fun, dy):
    #Guess mu and y_mu as the average of the x and y data
    y_mu_guess = float((sum(y))/(len(y)))*2
    mu_guess = float((sum(x))/(len(x)))
    #A good estimation for standard deviation is the span of your range of x values
    #divided by 4
    sigma_guess = float((max(x)-min(x))/2)

    param_guess = [mu_guess, sigma_guess, y_mu_guess]

    #Non-linear least squares data fit, outputs function parameters with uncertainties
    params, pcov = spop.curve_fit(f=fun, xdata=x, ydata=y, p0=param_guess, sigma=dy)
    #pcov is a 3x3 matrix who's diagonals correspond to the parameter uncertainties squared, so here
    #these uncertainties are extracted
    params_err = np.sqrt(np.diag(pcov))
    #Reduced chi-squared is now calculated using the fitted parameters
    chisq = red_chi_sq(x, y, fun, dy, params[0], params[1], params[2])

    return params, chisq, params_err 

#Plots all the various fits on the same plot
def fits_plot(x, y, dx, dy, p, dp, chisq, method):
    gauss_params = p[0]
    lognorm_params = p[1]
    laplace_params = p[2]

    gauss_params_err = dp[0]
    lognorm_params_err = dp[1]
    laplace_params_err = dp[2]

    gauss_chi_sq = str(round(chisq[0],2))
    lognorm_chi_sq = str(round(chisq[1],2))
    laplace_chi_sq = str(round(chisq[2],2))

    #if method == 'ls':
    #    gauss_params, gauss_params_err = param_swap(gauss_params, gauss_params_err)
    #    lognorm_params, lognorm_params_err = param_swap(lognorm_params, lognorm_params_err)
    #    laplace_params, laplace_params_err = param_swap(laplace_params, laplace_params_err)

    #Updating Matplotlib styling parameters
    plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.weight': 'book',
            'font.size': 8,
            'axes.labelsize': 9,
            'axes.titlesize': 14,
            'lines.linewidth': 1
        })
        
    #Creating figrue {Various Fits} (a bit janky)
    fig = plt.figure(figsize=(10,5))

    gs = gridspec.GridSpec(1, 1, hspace=0.5)
    plot = fig.add_subplot(gs[0,0])

    #Plotting data with error bars
    plot.errorbar(x, y, yerr=dy, xerr=dx, fmt='.k', capsize=2.5, label='Data')
    X = np.linspace(1, max(x)+10, 500)

    #Plotting gaussian/normal fit
    gau_mu = '$\mu$ = '+str(ez_round(gauss_params[0],gauss_params_err[0])[0])+' $\pm$ '+str(ez_round(gauss_params[0],gauss_params_err[0])[1])
    gau_sig = '$\sigma$ = '+str(ez_round(gauss_params[1],gauss_params_err[1])[0])+' $\pm$ '+str(ez_round(gauss_params[1],gauss_params_err[1])[1])
    gau_y_mu = '$y_{\\mu}$ = '+str(ez_round(gauss_params[2],gauss_params_err[2])[0])+' $\pm$ '+str(ez_round(gauss_params[2],gauss_params_err[2])[1]) 
    leg_gau_params = gau_y_mu+', '+gau_mu+', '+gau_sig
    plot.plot(X, gauss(X, gauss_params[0], gauss_params[1], gauss_params[2]), color='blue', 
               ls='--', label='Normal Fit, $\chi^2_{\\nu} = $'+gauss_chi_sq+'\n'+leg_gau_params)

    #Plotting lognormal fit
    log_mu = '$\mu$ = '+str(ez_round(lognorm_params[0],lognorm_params_err[0])[0])+' $\pm$ '+str(ez_round(lognorm_params[0],lognorm_params_err[0])[1]) 
    log_sig = '$\sigma$ = '+str(ez_round(lognorm_params[1],lognorm_params_err[1])[0])+' $\pm$ '+str(ez_round(lognorm_params[1],lognorm_params_err[1])[1])
    log_y_mu = '$y_{\\mu}$ = '+str(ez_round(lognorm_params[2],lognorm_params_err[2])[0])+' $\pm$ '+str(ez_round(lognorm_params[2],lognorm_params_err[2])[1])
    leg_log_params = log_y_mu+', '+log_mu+', '+log_sig
    plot.plot(X, lognormal(X, lognorm_params[0], lognorm_params[1], lognorm_params[2]), color='red', 
               ls='--', label='Lognormal Fit, $\chi^2_{\\nu} = $'+lognorm_chi_sq+'\n'+leg_log_params)
        
    #Plotting laplacian fit
    lap_mu = '$\mu$ = '+str(ez_round(laplace_params[0],laplace_params_err[0])[0])+' $\pm$ '+str(ez_round(laplace_params[0],laplace_params_err[0])[1])
    lap_sig = '$\sigma$ = '+str(ez_round(laplace_params[1],laplace_params_err[1])[0])+' $\pm$ '+str(ez_round(laplace_params[1],laplace_params_err[1])[1])
    lap_y_mu = '$y_{\\mu}$ = '+str(ez_round(laplace_params[2],laplace_params_err[2])[0])+' $\pm$ '+str(ez_round(laplace_params[2],laplace_params_err[2])[1]) 
    leg_lap_params = lap_y_mu+', '+lap_mu+', '+lap_sig
    plot.plot(X, laplacian(X, laplace_params[0], laplace_params[1], laplace_params[2]), color='green', 
               ls='--', label='Laplacian Fit, $\chi^2_{\\nu} = $'+laplace_chi_sq+'\n'+leg_lap_params)
    #Plot styling options
    plot.legend(fontsize=6)
    plot.set_xticks(np.arange(min(X)-1, max(X)+1, 5.0))
    plot.set_yticks(np.arange(0, max(y)+16, 20.0))
    plot.grid()
    plot.set_xlim([1,max(x)+10])
    plot.set_ylim([0,max(y)+20])
    plot.set_xlabel('x [scaled units]')
    plot.set_ylabel('y [scaled units]')
    plot.set_title('Various Fits on Ellipses Data')

    #Saving figures
    plt.savefig('Ellipses_Various_Fits_'+method, dpi=500)

#Plots the residuals for each fit
def resid_plot(x, y, dy, p, method, resid, dodr):
    gauss_params = p[0]
    lognorm_params = p[1]
    laplace_params = p[2]

    #Updating Matplotlib styling parameters
    plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.weight': 'book',
            'font.size': 8,
            'axes.labelsize': 9,
            'axes.titlesize': 14,
            'lines.linewidth': 1
        })
    
    #Creating figure {Fit Residuals}
    fig = plt.figure(figsize=(12,4))
    gs = gridspec.GridSpec(1, 3, hspace=0.3)
    X = np.linspace(1, max(x)+10, 500)

    #Gaussian/Normal fit residuals
    res1 = fig.add_subplot(gs[0,0])

    #Plotting residuals with error bars
    if method == 'ls':
        res1.errorbar(x, y - gauss(x, gauss_params[0], gauss_params[1], gauss_params[2]), yerr=dy, fmt='.k', capsize=2.5, label='Residuals')
    elif method == 'odr':
        res1.errorbar(x, resid[0], yerr=dodr[0], fmt='.k', capsize=2.5, label='Residuals')
    res1.plot(X, np.zeros_like(X), color = 'red', ls = '--')
    #Plot styling options
    res1.legend(fontsize=6)
    res1.set_xticks(np.arange(min(X)-1, max(X)+1, 5.0))
    res1.grid()
    res1.set_xlim([1,max(X)])
    res1.set_ylim([-15,15])
    res1.set_xlabel('x [scaled units]')
    res1.set_ylabel('$\delta$r [scaled units]')
    res1.set_title('Normal Fit Residuals')

    #Lognormal fit residuals
    res2 = fig.add_subplot(gs[0,1])

    #Plotting residuals with error bars
    if method == 'ls':
        res2.errorbar(x, y - lognormal(x, lognorm_params[0], lognorm_params[1], lognorm_params[2]), yerr=dy, fmt='.k', capsize=2.5, label='Residuals')
    elif method == 'odr':
        res2.errorbar(x, resid[1], yerr=dodr[1], fmt='.k', capsize=2.5, label='Residuals')
    res2.plot(X, np.zeros_like(X), color = 'red', ls = '--')
    #Plot styling options
    res2.legend(fontsize=6)
    res2.set_xticks(np.arange(min(X)-1, max(X)+1, 5.0))
    res2.grid()
    res2.set_xlim([1,max(X)])
    res2.set_ylim([-15,15])
    res2.set_xlabel('x [scaled units]')
    res2.set_ylabel('$\delta$r [scaled units]')
    res2.set_title('Lognormal Fit Residuals')

    #Laplacian fit residuals
    res3 = fig.add_subplot(gs[0,2])

    #Plotting residuals with error bars
    if method == 'ls':
        res3.errorbar(x, y - laplacian(x, laplace_params[0], laplace_params[1], laplace_params[2]), yerr=dy, fmt='.k', capsize=2.5, label='Residuals')
    elif method == 'odr':
        res3.errorbar(x, resid[2], yerr=dodr[2], fmt='.k', capsize=2.5, label='Residuals')
    res3.plot(X, np.zeros_like(X), color = 'red', ls = '--')
    #Plot styling options
    res3.legend(fontsize=6)
    res3.set_xticks(np.arange(min(X)-1, max(X)+1, 5.0))
    res3.grid()
    res3.set_xlim([1,max(X)])
    res3.set_ylim([-15,15])
    res3.set_xlabel('x [scaled units]')
    res3.set_ylabel('$\delta$r [scaled units]')
    res3.set_title('Laplacian Fit Residuals')

    #Saving figures
    plt.savefig('Ellipses_Residuals_'+method, dpi=500)

#Plots the normal probability for each fit
def normal_plot(x, y, dy, p, method):
    gauss_params = p[0]
    lognorm_params = p[1]
    laplace_params = p[2]

    #Updating Matplotlib styling parameters
    plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.weight': 'book',
            'font.size': 8,
            'axes.labelsize': 9,
            'axes.titlesize': 14,
            'lines.linewidth': 1
        })
    
    #Creating figure {Normal Probability}
    fig = plt.figure(figsize=(12,4))
    gs = gridspec.GridSpec(1, 3, hspace=0.3)
    X = np.linspace(1, max(x)+10, 500)
    Y = np.linspace(0, max(y)+10, 1000)

    #Gaussian/Normal fit normal probability
    nrm1 = fig.add_subplot(gs[0,0])

    #Plotting normal probability with error bars
    nrm1.errorbar(gauss(x, gauss_params[0], gauss_params[1], gauss_params[2]), y, yerr=dy, fmt='.k', capsize=2.5)
    nrm1.plot(Y, Y, color = 'red', ls = '--')
    #Plot styling options
    nrm1.set_xticks(np.arange(min(Y), max(Y)+1, 20.0))
    nrm1.set_yticks(np.arange(min(Y), max(Y)+1, 20.0))
    nrm1.grid()
    nrm1.set_xlim([min(Y),max(Y)])
    nrm1.set_ylim([min(Y),max(Y)])
    nrm1.set_xlabel('Theoretical Values [scaled units]')
    nrm1.set_ylabel('$Sample Values [scaled units]')
    nrm1.set_title('Normal Fit Normal Probability')

    #Lognormal fit normal probability
    nrm2 = fig.add_subplot(gs[0,1])

    #Plotting normal probability with error bars
    nrm2.errorbar(lognormal(x, lognorm_params[0], lognorm_params[1], lognorm_params[2]), y, yerr=dy, fmt='.k', capsize=2.5, label='Residuals')
    nrm2.plot(Y, Y, color = 'red', ls = '--')
    #Plot styling options
    nrm2.set_xticks(np.arange(min(Y), max(Y)+1, 20.0))
    nrm2.set_yticks(np.arange(min(Y), max(Y)+1, 20.0))
    nrm2.grid()
    nrm2.set_xlim([min(Y),max(Y)])
    nrm2.set_ylim([min(Y),max(Y)])
    nrm2.set_xlabel('Theoretical Values [scaled units]')
    nrm2.set_ylabel('$Sample Values [scaled units]')
    nrm2.set_title('Lognormal Fit Normal Probability')

    #Laplacian fit normal probability
    nrm3 = fig.add_subplot(gs[0,2])

    #Plotting normal probability with error bars
    nrm3.errorbar(laplacian(x, laplace_params[0], laplace_params[1], laplace_params[2]), y, yerr=dy, fmt='.k', capsize=2.5, label='Residuals')
    nrm3.plot(Y, Y, color = 'red', ls = '--')
    #Plot styling options
    nrm3.set_xticks(np.arange(min(Y), max(Y)+1, 20.0))
    nrm3.set_yticks(np.arange(min(Y), max(Y)+1, 20.0))
    nrm3.grid()
    nrm3.set_xlim([min(Y),max(Y)])
    nrm3.set_ylim([min(Y),max(Y)])
    nrm3.set_xlabel('Theoretical Values [scaled units]')
    nrm3.set_ylabel('$Sample Values [scaled units]')
    nrm3.set_title('Laplacian Fit Normal Probability')

    #Saving figures
    plt.savefig('Ellipses_Normals_'+method, dpi=500)

#Code taken from https://www.physics.utoronto.ca/apl/python/odr_fit_extended.py, gives MC CDF %
def monte_carlo(x, dx, dy, fun, p, chisq):
    #Helper function (needed for odr function, not important)
    def swapped(a,b):
        return fun(b,*a)
    
    Number_of_MC_iterations = 1000
    # Initialize Monte Carlo output distributions
    x_dist           = Number_of_MC_iterations*[None]
    y_dist           = Number_of_MC_iterations*[None]
    p_dist           = Number_of_MC_iterations*[None]
    quasi_chisq_dist = Number_of_MC_iterations*[None]
    # Initialize timing measurement
    import time
    start_time = time.process_time()
    for i in range(Number_of_MC_iterations) :
        # Starting with the x and x uncertainty (x_sigma) values from
        #    the data, calculate Monte Carlo values assuming a normal
        #    gaussian distibution.
        x_dist[i] = np.random.normal(x,dx)
        # Calculate y using Monte Carlo x, then smear by y uncertainty
        y_dist[i] = np.random.normal(fun(x,p[0], p[1], p[2]),dy)
        # Fit the Monte Carlo x,y pseudo-data to the original model
        data_dist = odr.RealData( x=x_dist[i], y=y_dist[i],
                                        sx=dx,  sy=dy)
        model_dist = odr.Model(swapped)
        fit_dist = odr.ODR(data_dist, model_dist, p, maxit=5000,
                                        job=10)
        output_dist = fit_dist.run()
        p_dist[i] = output_dist.beta
        quasi_chisq_dist[i] = output_dist.res_var
    end_time = time.process_time()

    print(f'simulations in {end_time-start_time} seconds.')
    # Sort the simulated quasi-chi-squared values
    quasi_chisq_sort = np.sort(quasi_chisq_dist)
    # Find the index of the sorted simulated quasi-chi-squared value
    # nearest to the data quasi-chi-squared value, to estimate the cdf
    CDF = 100.*(1.0-np.abs(quasi_chisq_sort-chisq).argmin()/float(Number_of_MC_iterations))

    return CDF

def __main__():
    #Loading data
    filename = 'Ellipses_Data.txt'
    data = load_data(filename)

    #Setting the calibration factor (confusingly labled scale factor in this code) and calculating
    #its uncertainty
    scale_f = round(float(25/13),2)
    dscale_f = scale_f * (0.5/13)

    x_data, x_data_err, y_data, y_data_err, s_data = data

    def scale_err(dx, dy):
        k = 0
        new_dx = [x + k for x in dx]
        new_dy = [y + k for y in dy]
        return new_dx, new_dy

    x_data_err, y_data_err = scale_err(x_data_err, y_data_err)

    #Scaling data values and doing error propogation calculations
    new_x, new_y, new_x_err, new_y_err = calc_uncertainty(x_data, y_data, x_data_err, y_data_err, s_data, scale_f, dscale_f)

    #Fit data using ODR
    gauss_params_odr, gauss_chisq_odr, gauss_params_err_odr, gauss_sigma_odr, gauss_resid_odr = odr_fit(new_x, new_y, gauss, new_x_err, new_y_err)
    logn_params_odr, logn_chisq_odr, logn_params_err_odr, logn_sigma_odr, logn_resid_odr = odr_fit(new_x, new_y, lognormal, new_x_err, new_y_err)
    lapl_params_odr, lapl_chisq_odr, lapl_params_err_odr, lapl_sigma_odr, lapl_resid_odr = odr_fit(new_x, new_y, laplacian, new_x_err, new_y_err)
    odr_params = [gauss_params_odr, logn_params_odr, lapl_params_odr]
    odr_chisq = [gauss_chisq_odr, logn_chisq_odr, lapl_chisq_odr]
    odr_params_err = [gauss_params_err_odr, logn_params_err_odr, lapl_params_err_odr]
    odr_resid = [gauss_resid_odr, logn_resid_odr, lapl_resid_odr]
    odr_sigma = [gauss_sigma_odr, logn_sigma_odr, lapl_sigma_odr]

    #Fit data using non-linear least-squares
    gauss_params_ls, gauss_chisq_ls, gauss_params_err_ls = nl_ls_fit(new_x, new_y, gauss, new_y_err)
    logn_params_ls, logn_chisq_ls, logn_params_err_ls = nl_ls_fit(new_x, new_y, lognormal, new_y_err)
    lapl_params_ls, lapl_chisq_ls, lapl_params_err_ls = nl_ls_fit(new_x, new_y, laplacian, new_y_err)
    ls_params = [gauss_params_ls, logn_params_ls, lapl_params_ls]
    ls_chisq = [gauss_chisq_ls, logn_chisq_ls, lapl_chisq_ls]
    ls_params_err = [gauss_params_err_ls, logn_params_err_ls, lapl_params_err_ls]

    #Plotting ODR fits and residuals and normals
    fits_plot(new_x, new_y, new_x_err, new_y_err, odr_params, odr_params_err, odr_chisq, 'odr')
    resid_plot(new_x, new_y, new_y_err, odr_params, 'odr', odr_resid, odr_sigma)
    normal_plot(new_x, new_y, new_y_err, odr_params, 'odr')

    #Plotting nl-ls fits and residuals and normals
    #fits_plot(new_x, new_y, new_x_err, new_y_err, ls_params, ls_params_err, ls_chisq, 'ls')
    #resid_plot(new_x, new_y, new_y_err, ls_params, 'ls', None, None)
    #normal_plot(new_x, new_y, new_y_err, ls_params, 'ls')

    #ODR CDF estimation and plotting
    gauss_CDF = monte_carlo(new_x, new_x_err, new_y_err, gauss, gauss_params_odr, gauss_chisq_odr)
    print(gauss_CDF)
    logn_CDF = monte_carlo(new_x, new_x_err, new_y_err, lognormal, logn_params_odr, logn_chisq_odr)
    print(logn_CDF)
    lapl_CDF = monte_carlo(new_x, new_x_err, new_y_err, laplacian, lapl_params_odr, lapl_chisq_odr)
    print(lapl_CDF)
       
if __name__ == '__main__':
    __main__()