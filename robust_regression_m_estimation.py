import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.tools import add_constant

def robust_regression(df, m_estimator):
    """ Fits a robust regression model on data through statsmodel.RLM() with spceified M-estimator.
    
    Args:
        df: pandas Dataframe containing attributes 'x' and 'y'.
        m_estimator: instance of statsmodel.robust.norms. Defines the M-estimator to use.
        
    Returns:
        result: a RegressionResult class instance obtained from fitting a robust regression.
    """
    model = sm.RLM(endog=df.y, exog=add_constant(df.x), M=m_estimator)
    result = model.fit()
    print(result.summary())
    return result

def run_robust_regression(data, m_estimator=sm.robust.norms.HuberT()):
    """ Fits robust regression models on all datasets in data. 
    
    Args:
        data: list of pandas Dataframe containing attributes 'x' and 'y'.
        m_estimator: instance of statsmodel.robust.norms. Defines the M-estimator to use.
        
    Returns:
        estimates: list of dictionaries containing parameter estimates for robust regression.
    """
    print('Using `m_estimator`:', m_estimator)
    estimates = []
    for idx, df in enumerate(data):
        print("Running robust regression on dataset %d" % (idx+1))
        result = robust_regression(df, m_estimator)
        params = {'a': result.params.x, 'b': result.params.const}
        estimates.append(params)
    return estimates

def robust_regression_weighted_data(df, weights, m_estimator=sm.robust.norms.HuberT()):
    """ Transform data to obtain (x', y') = (sqrt(weights) * x, sqrt(weights) * y). 
    Runs a robust regression on (x', y').
    
    Args:
        df: pandas Dataframe containing attributes 'x' and 'y'.
        weights: list of weights corresponding to each data point. 
        m_estimator: instance of statsmodel.robust.norms. Defines the M-estimator to use. Default to Huber. 
        
    Returns:
        result: a RegressionResult class instance obtained from fitting a robust regression.
        
    """
    y, x = df.y, df.x, 
    wls_model = sm.WLS(y, add_constant(x), weights=weights)
    yp, xp = wls_model.wendog, wls_model.wexog # Retrieve weighted data. 
    rlm_model = sm.RLM(yp, xp, M=m_estimator)
    result = rlm_model.fit()
    print(result.summary())
    return result

def run_robust_regression_weighted_data(data, weights_list, m_estimator=sm.robust.norms.HuberT()):
    """ Runs robust_regression_weighted_data() on each dataset in data. See that method's docstring for more details.
    Args:
        data: pandas Dataframe containing attributes 'x' and 'y'.
        weights_list: list of lists, where weights_list[i] corresponding to the list of weights for data[i].
        m_estimator: instance of statsmodel.robust.norms. Defines the M-estimator to use. Default to Huber. 
        
    Calls: robust_regression_weighted_data().
    
    Returns:
        result: a list of dictionary objects containing parameter estimates for each dataset. 
        
    """
    print('Using `m_estimator`:', m_estimator)
    estimates = []
    for idx, df in enumerate(data):
        print("Running robust regression on weighted data on dataset %d" % (idx+1))
        weights = weights_list[idx]
        print(len(weights))
        print(len(df))
        result = robust_regression_weighted_data(df, weights, m_estimator)
        print(result.params)
        params = {'a': result.params[1], 'b': result.params[0]}
        estimates.append(params)
    return estimates
