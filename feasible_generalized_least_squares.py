import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.tools import add_constant
from statsmodels.nonparametric.kernel_regression import KernelReg as kernelRegression

def feasible_generalized_least_squares(df, tol=1e-12, max_iters=100, use_log_residuals=True):
    """
    Runs feasible generalized least squares (FGLS) algorithm on a dataset to produce a weighted regression fit
    and estimate of the conditional variance of the errors. The reference for the algorithm is
    listed here: http://www.stat.cmu.edu/~cshalizi/uADA/12/lectures/ch06.pdf
    
    Args:
        df: pandas DataFrame containing data in 'x' and 'y' attributes. 
        tol: float value indicating when convergence is reached. 
        max_iters: maximum number of iterations of FGLS.
        use_log_residuals: boolean used in determining response variable of non-parametric regression in 
          estimating conditional variance at each iteration. If true, uses log squared residuals as response of
          non-parametric regression fit; if false, uses raw squared residuals. 
    
    Returns:
        A tuple of (regression, variance_estimate).
            regression: a RegressionResults class instance.
            variance_estimate: list containing estimated variance at each value of x in the dataset. 
    
    """
    i = 1 
    old_coefs = None
    eps = 1e-12
    
    if use_log_residuals:
        print("Using log-squared residuals method for estimating conditional mean of variance.")
    else:
        print("Using squared residuals method for estimating conditional mean of variance.")
    
    # First estimate regression parameters from an OLS estimate. 
    regression = sm.OLS(y, add_constant(x)).fit()
    coefs = regression.params
    
    # Repeat until parameter convergence (or we hit the iteration limit):
    while (old_coefs is None or (max(coefs - old_coefs) > tol) and (i <= max_iters)): 
        # Estimate the variance based on residuals. 
        if use_log_residuals: 
            log_squared_residuals = np.log(np.square(regression.resid)) 
            log_variance_estimates = kernelRegression(log_squared_residuals, x, var_type='c').fit()
            log_mean, log_marginals = log_variance_estimates
            mean = np.exp(log_mean) 
        else:
            squared_residuals = np.square(regression.resid)
            variance_estimates = kernelRegression(squared_residuals, x, var_type='c').fit()
            mean, marginals = variance_estimates
            
        # Fit a new weighted regression and update parameters. 
        old_coefs = coefs
        regression = sm.WLS(y, add_constant(x), weights = 1 / mean).fit()
        coefs = regression.params
        i += 1  
    return regression, mean 

        
def run_feasible_GLS(data, use_log_residuals=True, include_plots=False):
    """
    Runs feasible generalized least squares (FGLS) algorithm on all datasets. Returns list of dict's 
    containing parameter estimates.
    
    Args:
        data: list of pandas DataFrames containing datasets.
        use_log_residuals: boolean - see docstring of feasible_generalized_least_squares(). 
        include_plots: boolean - whether to include plots. 
    
    Returns:
        A list of dicts containing the parameter estimates for WLS. The list order is the dataset order.
    """
    regression_list = []
    weights_list = []
    estimates = []
    
    if include_plots:
        plt.figure(figsize=(10,10))
        
    for idx, df in enumerate(data):
        print("Running feasible GLS on dataset %d" % (idx + 1))
        result, weights = feasible_generalized_least_squares(df, use_log_residuals=use_log_residuals)
        print(result.summary())
        print("Plot of conditional variance estimates on this data set:")
        if include_plots:
            plt.scatter(df.x, weights, s=50, label='Dataset %d' % (idx+1))
        regression_list.append(result)
        weights_list.append(weights)
        params = {'a': result.params.x, 'b': result.params.const}
        estimates.append(params)
    if include_plots:
        plt.legend(loc=2, prop={'size': 12})
        plt.title('Estimates for conditional variance of errors')
        plt.ylabel('Estimated error variance')
        plt.xlabel('X')
        # plt.savefig('conditional_variance_estimates_raw.png')
    return estimates

def pooled_feasible_generalized_least_squares(data, 
                                              tol=1e-12,
                                              max_iters=5,
                                              use_log_residuals=True,
                                              include_plots=False):
    """
    Runs feasible generalized least squares (FGLS) algorithm on all datasets in data. In estimating the 
    conditional variance at each of the values of the independent variable, pools together all residuals and
    x values. See docstring of feasible_generalized_least_squares() for more information. 
    
    Args:
        data: list of pandas DataFrame containing data in 'x' and 'y' attributes. 
        tol: float value indicating when convergence is reached. 
        max_iters: maximum number of iterations of FGLS.
        use_log_residuals: boolean used in determining response variable of non-parametric regression in 
          estimating conditional variance at each iteration. If true, uses log squared residuals as response of
          non-parametric regression fit; if false, uses raw squared residuals. 
        include_plots: boolean - whether or not to output plots. 
    
    Returns:
        A tuple of (regression_list, variance_estimates_list).
            regression_list: a list of RegressionResults class instances.
            variance_estimates_list: a list of lists of conditional variances estimates at each x value in data.
              variance_estimates[0] is the list of conditional variance estimates at each point in the first dataset.
    
    """
    if include_plots:
        plt.figure(figsize=(12,12))
    i = 1
    eps = 1e-12
    
    if use_log_residuals:
        print("Using log-squared residuals method for estimating conditional mean of variance.")
    else:
        print("Using squared residuals method for estimating conditional mean of variance.")
    
    # Initialize placeholders. 
    old_coefs_list = [None] * len(data) 
    regression_list = [] 
    coefs_list = [] 
    variance_estimates_list = []
    
    # Perform initial regression fit on all data. 
    for df in data:
        x, y = df.x, df.y
        regression = sm.OLS(y, add_constant(x)).fit()
        coefs = regression.params
        
        regression_list.append(regression)
        coefs_list.append(coefs)
        variance_estimates_list.append(np.zeros(x.shape))
    
    # Turn lists into numpy lists for convenience. 
    regression_list = np.array(regression_list)
    coefs_list = np.array(coefs_list)
    old_coefs_list = np.array(old_coefs_list)
    
    init = True # `init` is True only on first iteration of while loop.
    while (init or (i <= max_iters)):
        if not init: # Check for convergence.
            deltas = [max(coef - old_coef) for coef, old_coef in zip(coefs_list, old_coefs_list)]
            if all(delta <= tol for delta in deltas):
                print("Stopping.")
                break
        # Pool together residuals and x values from all regressions.
        pooled_residuals = np.concatenate([reg.resid for reg in regression_list]).ravel()
        pooled_squared_residuals = np.square(pooled_residuals)
        pooled_x = np.concatenate([df.x.values for df in data]).ravel()
        
        # Estimate conditional variance. 
        if use_log_residuals:
            pooled_log_squared_residuals = np.log(pooled_squared_residuals + eps) # Add `eps` to prevent float('-inf'). 
            log_variance_estimates = kernelRegression(pooled_log_squared_residuals, pooled_x, var_type='c').fit()
            log_mean, log_marginals = log_variance_estimates
            mean = np.exp(log_mean)
        else:
            variance_estimates = kernelRegression(pooled_squared_residuals, pooled_x, var_type='c').fit()
            mean, marginals = variance_estimates
        
        label = 'Iteration %d ' % i 
        if include_plots:
            plt.scatter(pooled_x, mean, label=label), 
        old_coefs_list = coefs_list.copy()
        start_idx = 0 
        # Update regression fit for each dataset and update variables.
        for group_idx, df in enumerate(data): 
            x, y = df.x, df.y
            end_idx = start_idx + len(df)
            group_mean = mean[start_idx:end_idx]
            regression = sm.WLS(y, add_constant(x), weights = 1 / group_mean).fit()
            coefs = regression.params
            regression_list[group_idx] = regression
            coefs_list[group_idx] = coefs
            variance_estimates_list[group_idx] = group_mean
            start_idx = end_idx
        print("End of iteration %d" % i)
        i += 1  
        init = False
    if include_plots:    
        plt.legend()
        plt.title('Pooled estimates for conditional variance of errors using squared residuals method')
        plt.ylabel('Estimated error variance')
        plt.xlabel('X')
        # plt.savefig('conditional_variance_estimates.png')
    return regression_list, variance_estimates_list

def run_pooled_feasible_GLS(data, use_log_residuals=True, include_plots=False):
    """
    Runs pooled FGLS on the data and returns parameter estimates and variance estimates. 
    
    Calls:
        pooled_feasible_generalized_least_squares()
        
    Returns: 
        A tuple of (estimates, variance_estimates):
          estimates: a list of dicts containing the parameter estimates for WLS. The list order is the dataset order.
          variance_estimates: list of lists containing variance estimates at each data 'x' value. 
    """
    results, variance_estimates = pooled_feasible_generalized_least_squares(data,
                                                                            use_log_residuals=use_log_residuals,
                                                                            include_plots=include_plots)
    estimates = []
    for i in range(len(results)):
        result = results[i]
        print(result.summary())
        params = {'a': result.params.x, 'b': result.params.const}
        estimates.append(params)
    return estimates, variance_estimates

def analyze_weighted_least_squares(df, weights, include_plots=True):
    """
    Fits a weighted linear regression on data with specified weights. Returns weighted regression fit.
    
    Args:
        df: pandas DataFrame containing data in 'x' and 'y' attributes. 
        weights: list of weights for each data point in `df`. 
        include_plots: boolean - whether to output plots. 
        
    Returns:
        a statsmodel RegressionResult class instance.
    """
    x, y = df.x, df.y 
    
    wls_model = sm.WLS(y, add_constant(x), weights = weights)
    
    # NOTE: This is a hack in order to get statsmodel to print additional statistics as a result of the fit. 
    # We are defining a WLS model, but then fitting an OLS model with the transformed (weighted) data. 
    # Note that the `resid` attribute of this fit is the weighted residual. 
    results = sm.OLS(wls_model.wendog, wls_model.wexog).fit()
    
    df['weighted_residuals'] = results.resid
    df['weighted_squared_residuals'] = np.square(results.resid)
    
    influence = results.get_influence() # Get weighted regression residual summaries. 
    frame = influence.summary_frame() # Output this as a dataframe.

    df = pd.concat([df, frame], axis=1)
    if include_plots:
        fig, ax = plt.subplots(3,3, figsize=(20,20))
        # 1. Plot regression line with data points. 
        X_MIN, X_MAX = df.x.min(), df.x.max()
        xs = np.linspace(X_MIN, X_MAX, 50)
        b, a = results.params[0], results.params[1]
        ys_ols = xs * a + b 
        ax[0,0].plot(xs, ys_ols)
        ax[0,0].scatter(df.x, df.y)
        
        # 2. Weighted residuals vs explanatory variable.
        sns.regplot(x=df.x, y=df.standard_resid, ax=ax[0,1])\
        .set_title('Plot of WLS weighted standardized residuals')
        # 3. Weighted squared residuals vs. explanatory variable. 
        sns.regplot(data=df, x='x', y='weighted_squared_residuals', ax=ax[0,2], fit_reg=False)
        # 4. Leverage vs Squared Residual Plot
        sm.graphics.plot_leverage_resid2(results, ax=ax[1,0])
        # 5. Influence (measured in Cook's Distance) vs Leverage. 
        sm.graphics.influence_plot(results, ax=ax[1,1])
        # 6. QQ-plot of weighted standardized residuals against theoretical Normal distribution.
        sm.qqplot(df.standard_resid, line='45', ax=ax[1,2])
        # 7. Cook's distance of each data point by index.
        df.plot(y='cooks_d', style='o', use_index=True, ax=ax[2,0])
        # 8. Studentized residual of each data point by index. 
        df.plot(y='student_resid', style='o', use_index=True, ax=ax[2,1])
    
        # Save the full figure...
#       fig.savefig('full_figure.png')

        # Save subplot.
        # extent = ax[1,2].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        # fig.savefig('qq_plot_weighted_residuals.png', bbox_inches=extent.expanded(1.1, 1.2))
    return results    
