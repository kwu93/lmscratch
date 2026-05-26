import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.tools import add_constant

def ordinary_least_squares(df, include_plots=True):
    """ Fits a regression model via ordinary least squares and summarizes results.
    Args:
        df: pandas DataFrame containing data in 'x' and 'y' attributes. 
        include_plots: boolean value; if true, output plots of data, residuals, squared residuals,
          influential points and outliers, among others. 

    Returns:
        A statsmodel RegressionResults class instance.
    """
    
    # Define an ordinary least squares model on data. 
    model = sm.OLS(endog=df.y, exog=add_constant(df.x))

    # Fit this model (using robust heteroscedasticity-consistent (HC3) standard errors for reporting).
    results = model.fit(cov_type='HC3') 
    print("OLS Summary:", results.summary())
    
    # Attach OLS influence metrics (Cook's distance, studentized residuals) to dataframe.
    influence = OLSInfluence(results)
    frame = influence.summary_frame()
    df = pd.concat([df, frame], axis=1)
    
    # Add columns for plotting convenience. 
    df['ols_residuals'] = results.resid
    df['ols_squared_residuals'] = results.resid**2
    df['ols_fitted'] = results.fittedvalues
    
    if include_plots:
        fig, ax = plt.subplots(3,3, figsize=(20,20))
        # 1. Scatterplot with regression line.
        sns.regplot(data=df, x='x', y='y', ax = ax[0,0], fit_reg=True)    
        # 2. Standardized residuals vs. explanatory variable.
        sns.regplot(data=df, x='x', y='standard_resid', ax = ax[0,1])
        # 3. Squared residuals vs. explanatory variable. 
        sns.regplot(data=df, x='x', y='ols_squared_residuals', ax=ax[0,2], fit_reg=False)
        # 4. Leverage vs Squared Residual Plot
        sm.graphics.plot_leverage_resid2(results, ax=ax[1,0])
        # 5. Influence (measured in Cook's Distance) vs Leverage. 
        sm.graphics.influence_plot(results, ax=ax[1,1])
        # 6. QQ-plot of standardized residuals against theoretical Normal distribution.
        sm.qqplot(df.standard_resid, line='45', ax=ax[1,2])
        # 7. Cook's distance of each data point by index.
        df.plot(y='cooks_d', style='o', use_index=True, ax=ax[2,0])
        # 8. Studentized residual of each data point by index. 
        df.plot(y='student_resid', style='o', use_index=True, ax=ax[2,1])
        # 9. DFBETA plots: influence of regression parameters with or without data point.
        df.plot(y='dfb_const', style='o', use_index=True, ax=ax[2,2], ylim=(-1,1))
        df.plot(y='dfb_x', style='o', use_index=True, ax=ax[2,2], ylim=(-1,1))
        
        # Save the full figure...
        # fig.savefig('full_figure.png')

        # Save a particular subplot. 
        # extent = ax[1,2].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        # fig.savefig('qq_plot_ols_residuals.png', bbox_inches=extent.expanded(1.1, 1.2))
        
    return results

def run_OLS(data, include_plots=False):
    """ Iterates over all datasets in data and performs an ordinary least squares estimate on the dataset.
    
    Calls:
        ordinary_least_squares()
    
    Args:
        data: list of pandas DataFrames containing datasets. 
        include_plots: boolean value; if true, output plots of data, residuals, diagnostics, etc. 
    
    Returns:
        A list of dicts containing the parameter estimates for OLS. The list order is the dataset order.
    """
    estimates = []
    for idx, df in enumerate(data):
        print("Running OLS on dataset %d" % (idx + 1))
        result = ordinary_least_squares(df, include_plots=include_plots)
        params = {'a': result.params.x, 'b': result.params.const}
        estimates.append(params)
    return estimates
