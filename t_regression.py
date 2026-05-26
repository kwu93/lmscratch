import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pymc3 as pm 

def run_bayesian_regression(df, error_distribution=pm.glm.families.StudentT()):
    """ 
    Runs a Bayesian linear regression with specified `error distribution` using PyMC3. 
    Outputs plots for posterior distribution of parameters, and posterior_predictive lines. 
    
    Args:
        df: pandas Dataframe containing data with 'x' and 'y' attributes.
        error_distribution: Instance of PyMC3.glm.families classes. Specifies error distribution of regression.
        
    Returns:
        None.
    """

    with pm.Model() as model_robust:
        pm.GLM.from_formula('y~x', df, family=error_distribution)
        trace = pm.sample(5000)
        print(pm.summary(trace))
        
        pm.plots.traceplot(trace)
        plt.figure(figsize=(12,12))
        plt.plot(df.x, df.y, 'x')
        pm.plots.plot_posterior_predictive_glm(trace, samples=100, eval=np.linspace(df.x.min(), df.x.max(), 100), label='posterior predictive regression lines')
        plt.legend()  
