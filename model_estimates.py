import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels as sm

from ordinary_least_squares import ordinary_least_squares, run_OLS
from feasible_generalized_least_squares import pooled_feasible_generalized_least_squares, run_pooled_feasible_GLS
from robust_regression_m_estimation import robust_regression, run_robust_regression, \
                                        robust_regression_weighted_data, run_robust_regression_weighted_data

def run_all_models(data):
    """ Runs all models and stores parameter estimates from each model in dictionary and returns it. 
    
    Args:
      data: list of pandas DataFrames containing data. 
      
    Returns:
      model_estimates: dictionary containing dictionary of parameter estimates.
        Example: model_estimates['ols']['a'] returns slope from OLS estimate. 
                 model_estimates['ols']['b'] returns intercept from OLS estimate. 
    """
    model_estimates = {}
    model_estimates['ols'] = run_OLS(data)
    model_estimates['pooled_wls'], variance_estimates = run_pooled_feasible_GLS(data, use_log_residuals=True)
    weights = [1.0 / var_est for var_est in variance_estimates]
    model_estimates['robust_huber'] = run_robust_regression(data, m_estimator=sm.robust.norms.HuberT())
    model_estimates['robust_tukey'] = run_robust_regression(data, m_estimator=sm.robust.norms.TukeyBiweight())
    model_estimates['wls_robust_huber'] = run_robust_regression_weighted_data(data, 
                                                                              weights,
                                                                              m_estimator=sm.robust.norms.HuberT())
    model_estimates['wls_robust_tukey'] = run_robust_regression_weighted_data(data, 
                                                                              weights,
                                                                              m_estimator=sm.robust.norms.TukeyBiweight())
    return model_estimates

def plot_data_and_models(data, model_estimates):
    """ Plots regression line for each model from the `model_estimates` dictionary overlayed on top of data.
    These plots are generated for each dataset in data. 
    
    Args:
      data: list of pandas DataFrames containing data. 
      model_estimates: dictionary containing dictionary of parameter estimates. See run_all_models() for details.
      
    Returns:
        None.
    """
    nrows = 3
    ncols = 2
    X_MIN, X_MAX = -25, 25
    fig, ax = plt.subplots(nrows, ncols, figsize = (20,20))

    for idx, df in enumerate(data):
        i,j = int(idx / ncols), idx % ncols
        # Scatter plot data with black dots. 
        sns.regplot(data=df, x='x', y='y', ax = ax[i,j], fit_reg=False, scatter_kws={"color": "black"})
        
        # Setup evenly spaced x values for interpolating regression fit for each model. 
        X_MIN, X_MAX = df.x.min(), df.x.max()
        xs = np.linspace(X_MIN, X_MAX, 50)
        
        # Produce estimates for mean y value given x.
        ys_ols = xs * model_estimates['ols'][idx]['a'] + model_estimates['ols'][idx]['b']
        ys_wls_pooled = xs * model_estimates['pooled_wls'][idx]['a'] + model_estimates['pooled_wls'][idx]['b']
        ys_robust_huber = xs * model_estimates['robust_huber'][idx]['a'] + model_estimates['robust_huber'][idx]['b']
        ys_robust_tukey = xs * model_estimates['robust_tukey'][idx]['a'] + model_estimates['robust_tukey'][idx]['b']
        ys_wls_robust_huber = xs * model_estimates['wls_robust_huber'][idx]['a'] + model_estimates['wls_robust_huber'][idx]['b']
        ys_wls_robust_tukey = xs * model_estimates['wls_robust_tukey'][idx]['a'] + model_estimates['wls_robust_tukey'][idx]['b']
        
        # Plot regression lines on same plot. 
        ax[i,j].plot(xs, ys_ols, label='ols')
        ax[i,j].plot(xs, ys_wls_pooled, label='pooled_wls')
        ax[i,j].plot(xs, ys_robust_huber, label='robust_huber')
        ax[i,j].plot(xs, ys_robust_tukey, label='robust_tukey')
        ax[i,j].plot(xs, ys_wls_robust_huber, label='wls_robust_huber')
        ax[i,j].plot(xs, ys_wls_robust_tukey, label='wls_robust_tukey')
        ax[i,j].legend()
        
        # Save each subplot. 
        extent = ax[i,j].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        SAVE_FILE = 'result_plot_%d.eps' % (idx + 1)
        fig.savefig(SAVE_FILE, bbox_inches=extent.expanded(1.1, 1.2), format='eps', dpi=1000)