#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""

pyPortfolioAnalysis: Methods to optimize portfolio
==================================================

Documentation is available as docstring or as HTML on https://github.com/anuragagrawaal/pyPortfolioAnalysis

Fucntions and Classes 



Optimization Methods
====================

 'HHI'
 'VaR'
 'VaR_portfolio'
 'add_constraint'
 'add_objective'
 'black_litterman'
 'box_constraint'
 'cVaR_portfolio'
 'constrained_objective'
 'diversification'
 'diversification_constraint'
 'equal_weight'
 'extract_groups'
 'extract_objective_measure'
 'extract_weights'
 'factor_exposure_constraint'
 'fn_map'
 'generate_sequence'
 'get_constraints'
 'group_constraint'
 'group_fail'
 'inverse_volatility_weights'
 'leverage_exposure_constraint'
 'leverage_fail'
 'max_sum_fail'
 'min_sum_fail'
 'minmax_objective'
 'normalize_weights'
 'optimize_portfolio'
 'performance_metrics_objective
 'port_mean'
 'portfolio_risk_objective'
 'portfolio_spec'
 'pos_limit_fail'
 'position_limit_constraint'
 'return_constraint'
 'return_objective'
 'risk_budget_objective'
 'rp_decrease'
 'rp_decrease_leverage'
 'rp_increase'
 'rp_position_limit'
 'rp_transform'
 'transaction_cost_constraint'
 'turnover'
 'turnover_constraint'
 'turnover_objective'
 'var_portfolio'
 'weight_concentration_objective'
 'weight_sum_constraint'


Plots
=====
 'chart_efficient_frontier'
 'chart_group_weights'
 'chart_weights'




References
----------
Brian G. Peterson and Peter Carl (2018). PortfolioAnalytics: Portfolio Analysis, Including Numerical Methods for Optimization of Portfolios. R package version 1.1.0. https://CRAN.R-project.org/package=PortfolioAnalytics
  
Boudt, Kris and Lu, Wanbo and Peeters, Benedict, Higher Order Comoments of Multifactor Models and Asset Allocation (June 16, 2014). Available at SSRN: http://ssrn.com/abstract=2409603 or http://dx.doi.org/10.2139/ssrn.2409603

Chriss, Neil A and Almgren, Robert, Portfolios from Sorts (April 27, 2005). Available at SSRN: http://ssrn.com/abstract=720041 or http://dx.doi.org/10.2139/ssrn.720041

Meucci, Attilio, The Black-Litterman Approach: Original Model and Extensions (August 1, 2008). Shorter version in, THE ENCYCLOPEDIA OF QUANTITATIVE FINANCE, Wiley, 2010. Avail- able at SSRN: http://ssrn.com/abstract=1117574 or http://dx.doi.org/10.2139/ssrn.1117574

Meucci, Attilio, Fully Flexible Views: Theory and Practice (August 8, 2008). Fully Flexible Views: Theory and Practice, Risk, Vol. 21, No. 10, pp. 97-102, October 2008. Available at SSRN: http://ssrn.com/abstract=1213325

Scherer, Bernd and Martin, Doug, Modern Portfolio Optimization. Springer. 2005.

Shaw, William Thornton, Portfolio Optimization for VAR, CVaR, Omega and Utility with General Return Distributions: A Monte Carlo Approach for Long-Only and Bounded Short Portfolios with Optional Robustness and a Simplified Approach to Covariance Matching (June 1, 2011). Available at SSRN: http://ssrn.com/abstract=1856476 or http://dx.doi.org/10.2139/ssrn.1856476

"""


# In[2]:


class black_litterman:
    
    """
    A class to call black_litterman object.
    
    Black-litterman formula is popular to get posterior moments of the portfolio.

    ...

    Attributes
    ----------
    R : pandas.DataFrame
        dataframe of returns series.
    P : matrix-like,
        KXN link matrix where N is the number of assets and K are the views
        
    Methods
    -------
    fit(Mu = None, S = None, Views = None, tau = 1):
        returns a matrix of posterior returns based on Views and returns data.
    """
    def __init__(self, R, P):
        
        """
    Constructor for all necessary attributes of class black_litterman

    ...

    Attributes
    ----------
    R : pandas.DataFrame
        dataframe of returns series.
    P : matrix-like,
        KXN link matrix where N is the number of assets and K are the views
    """
        self.R = R
        self.P = P
        
    def fit(self, Mu = None, S = None, Views = None, tau = 1):
        
        """
    fit method of black_litterman object
    
    Main function to call in order to return the posterior mean of the portfolio given the views and return.
    
    Parameters
    ----------
    Mu : array-like, optional
        prior mean of the returns. numpy.mean is called if Mu is None
    S : matrix-like, optional
        NXN covariance matrix. numpy.cov is called if S is None
    Views: array-like, default = None
        array of K views held by investor. 
    Tau: float, default = 1
        multiplying factor to be used.
        
    Returns
    -------
    array:
        returns an array of posterior moments of portfolio given the returns and weights
    
    See Also
    --------
    portfolio_spec
    optimize_portfolio
    
    Notes
    -----
    
    Black-litterman formula is very useful if an investor believes that a particular stock is going to rise
    compared to others although this information is not incorporated in the stock price. Essentially, it captures
    the views an investor holds for a particular asset.
    
    Examples
    --------
    >>> import pandas_datareader as pdr
    >>> aapl = pdr.get_data_yahoo('AAPL')
    >>> msft = pdr.get_data_yahoo('MSFT')
    >>> tsla = pdr.get_data_yahoo('TSLA')
    >>> port = pd.DataFrame({'aapl': pd.DataFrame.reset_index(aapl).iloc[:,6], 'msft':pd.DataFrame.reset_index(msft).iloc[:,6],
                       'tsla': pd.DataFrame.reset_index(tsla).iloc[:,6]})
    >>> port_ret = port.pct_change().dropna()
    >>> P = np.matrix([[-1, 0, 1],[0,-1,1]])
    >>> Views = np.matrix([[0.02],
                   [0.05]])
    >>> bl = black_litterman(R = port_ret, P = P)
    >>> bl.fit(Views = Views)

    """
        import numpy as np
        import pandas as pd
        import math as math
        import random
    


        if Mu == None:
            Mu = np.transpose( np.asmatrix(pd.DataFrame.mean(self.R)))
        if len(self.R.columns) != len(Mu):
            raise SystemExit('Length of Mu must equal R')
        if S == None:
            S = self.R.cov()
        if len(self.R.cov()) & len(np.transpose(self.R.cov())) != len(self.R.columns):
            raise SystemExit('Dimensions of Sigma must equal Dimensions of R')
        Omega = tau * self.P.dot(self.R.cov()).dot(np.transpose(self.P))
        if np.any(Views) == None:
            Views = np.diagonal(Omega)
        tot_wgt = np.linalg.inv((tau * np.linalg.inv(S))+np.transpose(self.P).dot(np.linalg.inv(Omega)).dot(self.P))
        wgt_avg = (tau * np.linalg.inv(S)).dot(Mu) + np.transpose(self.P).dot(np.linalg.inv(Omega)).dot(Views)
        return(tot_wgt.dot(wgt_avg))


# In[3]:


class portfolio_spec:
    
    """
    A class to call portfolio_spec object.
    
    portfolio_spec object is the main class that contains all the constraints, objectives, optimal weights;
    it is called by many functions to get relevant attributes and store attributes.

    ...

    Attributes
    ----------
    
    assets : int, or array-like,
        add assets to portfolio either via name of assets in the form of an array or int of number of assets.
    category_label: dict, optional
        dictionary of different categories assigned to different assets. similar to group_constraint.
        See group_constraint.
    weights_seq : sequence, optional
        sequence of random weights. These weights will be used to optimize weights. See generate_sequence
    message : bool, default = False
        bool to enable or diable message
        
    Methods
    -------
    port_summary():
        returns a dictionary of the summary of constraints and objective added by add_consraint and add_objective
        funtions. See add_constraint, add_objective
    optimal_portfolio():
        returns a dictionary of optimal weights, objective measure and minimum value calculated by the specified solver
        NOTE: optimal_portfolio will only return result if optimize_portfolio is called. See optimize_portfolio
    """
    
    def __init__(self, assets, category_labels = None, weights_seq = None, message = False):
        
        """
    Constructor for all necessary attributes of class black_litterman

    ...

    Attributes
    ----------
    
    assets : int, or array-like,
        add assets to portfolio either via name of assets in the form of an array or int of number of assets.
    category_label: dict, optional
        dictionary of different categories assigned to different assets. similar to group_constraint.
        See group_constraint.
    weights_seq : sequence, optional
        sequence of random weights. These weights will be used to optimize weights. See generate_sequence
    message : bool, default = False
        bool to enable or diable message

    """
        import numpy as np
        import pandas as pd




        if assets == None:
            raise SystemExit('Please Enter Asset Names')
        if type(assets) == int:
            assetnames = ['asset']*assets
            for i in range(0, assets):
                assetnames[i] = ''.join(['asset', '.', str(i+1)])
                assets = assetnames
        if type(assets) == list or type(assets) == np.ndarray:
            assetnames = assets
        if np.any(category_labels) == None:
            self.category_labels = None
        if not(np.any(category_labels) == None):
            tmp_cat_len = []
            for i in range(0, len(category_labels)):
                tmp_cat_len.append(len(list(category_labels.values())[i]))
            tmp_cat_len = sum(tmp_cat_len)
            if len(assets) != tmp_cat_len:
                raise SystemExit('len(assets) must equal len(category_labels)')
            if type(category_labels) != dict:
                raise SystemExit('Category_labels must be a list')
        self.category_labels = category_labels
        self.assets = assets
        self.nassets = len(assets)
        self.constraints = None
        self.objectives = None
        self.weight_seq = None
        self.weights = None
        self.objective_measures = None
    def port_summary(self):
        
        """
    summary of the portfolio
    
    method to provide a summary of the portfolio.
            
    Returns
    -------
    dict:
        returns a dictionary of all constraints and objective
    
    See Also
    --------
    portfolio_spec
    optimize_portfolio
    
    Examples
    --------
    >>> import pandas_datareader as pdr
    >>> aapl = pdr.get_data_yahoo('AAPL')
    >>> msft = pdr.get_data_yahoo('MSFT')
    >>> tsla = pdr.get_data_yahoo('TSLA')
    >>> uber = pdr.get_data_yahoo('UBER')
    >>> amzn = pdr.get_data_yahoo('AMZN')
    >>> port = pd.DataFrame({'aapl': pd.DataFrame.reset_index(aapl).iloc[:,6], 'msft':pd.DataFrame.reset_index(msft).iloc[:,6],
        'tsla': pd.DataFrame.reset_index(tsla).iloc[:,6], 'uber':pd.DataFrame.reset_index(uber).iloc[:,6],
        'amzn': pd.DataFrame.reset_index(amzn).iloc[:,6]})
    >>> port_ret = port.pct_change().dropna()
    >>> R = port_ret
    >>> p1 = portfolio_spec(assets = 3)
    >>> add_constraint(portfolio = p1, kind = 'factor_exposure', B = [0.2,0.2,0.4], lower = 1.0, upper = 0.9)
    >>> add_constraint(portfolio = p1, kind = 'group', groups = dict(zip(['equity','debt'],[[0,1], [2]])), group_min = 0.2, group_max = 0.5)
    >>> add_constraint(portfolio = p1, kind = 'transaction', ptc = 0.2)
    >>> add_objective(portfolio = p1, kind = 'return', target = 0.1, name = 'return_obj')
    >>> add_objective(portfolio = p1, kind = 'minmax',  minimum = 0.2, maximum = 0.3,name = 'risk')
    >>> p1.port_summary()
    """
        return({'Assets':self.assets,'Number of Assets':self.nassets,'category_labels':self.category_labels,
              'Constraints':self.constraints, 'Objectives':self.objectives})
    def optimal_portfolio(self):
        
        """
    summary of the portfolio
    
    method to provide a summary of the optimal weights, objective and minimum output of portfolio.
            
    Returns
    -------
    dict:
        returns a dictionary of optimal weights, objective and minimum output of portfolio
    
    See Also
    --------
    portfolio_spec
    optimize_portfolio
    
    Examples
    --------
    >>> import pandas_datareader as pdr
    >>> aapl = pdr.get_data_yahoo('AAPL')
    >>> msft = pdr.get_data_yahoo('MSFT')
    >>> tsla = pdr.get_data_yahoo('TSLA')
    >>> uber = pdr.get_data_yahoo('UBER')
    >>> amzn = pdr.get_data_yahoo('AMZN')
    >>> port = pd.DataFrame({'aapl': pd.DataFrame.reset_index(aapl).iloc[:,6], 'msft':pd.DataFrame.reset_index(msft).iloc[:,6],
        'tsla': pd.DataFrame.reset_index(tsla).iloc[:,6], 'uber':pd.DataFrame.reset_index(uber).iloc[:,6],
        'amzn': pd.DataFrame.reset_index(amzn).iloc[:,6]})
    >>> port_ret = port.pct_change().dropna()
    >>> R = port_ret
    >>> p1 = portfolio_spec(assets = 3)
    >>> add_constraint(portfolio = p1, kind = 'factor_exposure', B = [0.2,0.2,0.4], lower = 1.0, upper = 0.9)
    >>> add_constraint(portfolio = p1, kind = 'group', groups = dict(zip(['equity','debt'],[[0,1], [2]])), group_min = 0.2, group_max = 0.5)
    >>> add_constraint(portfolio = p1, kind = 'transaction', ptc = 0.2)
    >>> add_objective(portfolio = p1, kind = 'return', target = 0.1, name = 'return_obj')
    >>> add_objective(portfolio = p1, kind = 'minmax',  minimum = 0.2, maximum = 0.3,name = 'risk')
    >>> optimize_portfolio(R, p1, optimize_method = 'DEoptim')
    >>> p1.optimal_weights()
    """
        import numpy as np
        import pandas as pd
        import math as math
        import random




        if np.any(self.weights) == None:
            raise SystemExit('Please run optimize_portfolio function before call optimzal weights')
        return({'Assets':self.assets,'Number of Assets':self.nassets,'category_labels':self.category_labels,
               'Weights':self.weights, 'Objective_measures':self.objective_measures})


# In[4]:


def add_constraint(portfolio, kind, enabled = True, **kwargs):
    """
    Add a constraint in portfolio_spec object
    
    Main function to add or update constraint in portfolio_spec object
    
    Parameters
    ----------
    portfolio : portfolio_spec,
            an object of class portfolio_spec. see portfolio_spec
    kind : str,
            currently supported constraint: ’weight_sum’ (also ’leverage’ or ’weight’), ’box’, ’group’,
           ’turnover’,’diversification’, ’position_limit’, ’return’, ’factor_exposure’,
            or ’leverage_exposure’.    
    enabled : bool, default = True
            bool to enable or disable constraints.
    kwargs : additional key word arguments, optional
            any additional constraint argument to be passed.
    
    Returns
    -------
    Adds constraints to object portfolio_spec of the specified input. 
    
    See Also
    --------
    add_constraints
    box_constraints
    group_constraints
    weight_sum_constraint
    turnover_constraint
    diversification_constraint
    position_limit_constraint
    return_constraint
    factor_exposure_constraint
    transaction_cost_constraint
    leverage_exposure_constraint

    
    Notes
    -----
    •weight_sum, weight, leverage Specify constraint on the sum of the weights, see weight_sum_constraint    
    •full_investment Special case to set min_sum=1 and max_sum=1 of weight sum constraints
    •dollar_neutral, active Special case to set min_sum=0 and max_sum=0 of weight_sum constraints
    •box box constraint for the individual asset weights, see box_constraint
    •long_only Special case to set min=0 and max=1 of box constraint
    •group specify the sum of weights within groups and the number of assets with non-zero weights in groups,
        see group_constraint
    •turnover Specify a constraint for target turnover. Turnover is calculated from a set of initial weights,
        see turnover_constraint
    •diversification target diversification of asset of weights, see diversification_constraint
    •position_limit Specify the number of non-zero,long, and/orshortpositions, see position_limit_constraint
    •return Specify the target mean return, see return_constraint
    •factor_exposure Specify risk factor exposures, see factor_exposure_constraint
    •leverage_exposure Specify a maximum leverage exposure, see leverage_exposure_constraint
    
    Examples
    --------
    >>> portfolio = portfolio_spec(assets = 4)
    >>> # adding weight_sum cinstraint
    >>> add_constraint(portfolio, kind = 'weight_sum', min_sum = 0.9, max_sum = .95)
    >>> # long_only is a special kind of box constraint where minimum and maximum is positive
    >>> # minimum and maximum can be list or scalars
    >>> add_constraint(portfolio, kind = 'box', minimum = [0.9, -0.5,-0.5, 0.1], maximum = 1)
    >>> add_constraint(portfolio, kind = 'long_only')

    >>> # group constraint is used to specify min and max weights of certain asset_groups
    >>> # groups must be a dict and group_min and group_max can be scalar or list
    >>> add_constraint(portfolio, kind = 'group', group_min = [0.9, -0.5,-0.5, 0.1], group_max = 1,
                  groups = {'eqity':[0,3], 'debt':[1,2]})

    >>> # adding weight_sum constraint
    >>> add_constraint(portfolio, kind = 'weight_sum', min_sum = 0.9, max_sum = .95)
    >>> # special case of weight_sum is dollar_neutral/active or full_investment
    >>> add_constraint(portfolio, kind = 'dollar_neutral')
    >>> add_constraint(portfolio, kind = 'full_investment')

    >>> #turnover constraint
    >>> add_constraint(portfolio, kind = 'turnover', turnover_target = 0.1)

    >>> #diversification constraint for diversification target in a portfolio
    >>> add_constraint(portfolio, kind = 'diversification', div_target = 0.1)


    >>> #position_limit is a constraint to restrict max position and also max long/short positions
    >>> add_constraint(portfolio, kind = 'position_limit', max_pos = 3, max_pos_long = 2, max_pos_short = 2)

    >>> #return constraint to add a target mean historical return to the portfolio
    >>> add_constraint(portfolio, kind = 'return', return_target = 0.0018)

    >>> #adds a tranaction cost constraint on portfolio
    >>> add_constraint(portfolio, kind = 'transaction_cost', ptc = 0.1)

    >>> #constraint on the leverage of portfolio
    >>> add_constraint(portfolio, kind = 'leverage_exposure', leverage = 0.8)

    >>> #Factor_exposure constraint is used to test portfolio with their impact on certaint factors
    >>> # factors can be single or multiple with lower and upper limit. 
    >>> # B can be a N*K matrix for N assets and K factors.
    >>> # lower and upper arguments can be float for single factor and list for multiple factors
    >>> add_constraint(portfolio, kind = 'factor_exposure', 
                  B = np.matrix([[1.2,1.3],
                                 [2.3,1.4],
                                 [1.4,0.9],
                                 [3.4,1.2]]),
                   lower = [0.8,1.4], upper = [2,2.4])
"""
    import numpy as np
    import pandas as pd

    import math as math
    import random




    
    if kind == None:
        raise ValueError('You must supply kind of portfolio')
    assets = portfolio.assets
    tmp_constraint = None
    if kind == 'box':
        tmp_constraint = box_constraint(assets = assets, kind = kind, enabled = enabled, message = True, **kwargs)
    elif kind == 'long_only':
        tmp_constraint = box_constraint(assets = assets, kind = kind, enabled = enabled, message = True, minimum = 0, maximum = 1, **kwargs)
    elif kind == 'group':
        tmp_constraint = group_constraint(assets = assets, kind = kind, enabled = enabled, message = True, **kwargs)
    elif kind == 'weight' or kind == 'leverage' or kind == 'weight_sum':
        tmp_constraint = weight_sum_constraint(assets = assets, kind = kind, enabled = enabled, message = True, **kwargs)
    elif kind == 'full_investment':
        tmp_constraint = weight_sum_constraint(assets = assets, kind = kind, enabled = enabled, message = True, min_sum = 1, max_sum = 1, **kwargs)
    elif kind == 'dollar_neutral':
        tmp_constraint = weight_sum_constraint(assets = assets, kind = kind, enabled = enabled, message = True, min_sum = 0, max_sum = 0,**kwargs)
    elif kind == 'turnover':
        tmp_constraint = turnover_constraint(assets = assets, kind = kind, enabled = enabled, message = True, **kwargs)
    elif kind == 'diversification':
        tmp_constraint = diversification_constraint(assets = assets, kind = kind, enabled = enabled, message = True, **kwargs)
    elif kind == 'position_limit':
        tmp_constraint = position_limit_constraint(assets = assets, kind = kind, enabled = enabled, message = True, **kwargs)
    elif kind == 'return':
        tmp_constraint = return_constraint(assets = assets, kind = kind, enabled = enabled, message = True, **kwargs)
    elif kind == 'factor_exposure':
        tmp_constraint = factor_exposure_constraint(assets = assets, kind = kind, enabled = enabled, message = True, **kwargs)
    elif kind == 'transaction':
        tmp_constraint = transaction_cost_constraint(assets = assets, kind = kind, enabled = enabled, message = True, **kwargs)
    elif kind == 'leverage_exposure':
        tmp_constraint = leverage_exposure_constraint(assets = assets, kind = kind, enabled = enabled, message = True, **kwargs)
    else:
        return(portfolio)
    if portfolio.constraints == None:
        portfolio.constraints = dict()
        portfolio.constraints[kind] = tmp_constraint
    else:
        portfolio.constraints[kind] = tmp_constraint
    return(portfolio)


# In[5]:


def box_constraint(assets, minimum, maximum, kind = 'box', enabled = True, message = True, **kwargs):
    """
    
    The box constraint defines the upper and lower limits of asset weights.
    add_constraint calls this function when type=”box” is defined.
    
    Parameters
    ----------
    assets :  int, or array-like,
              number of assets, or optionally a named list of assets specifying initial weights.
    minimum : float, or array-like,
              numeric or named list defining the minimum constraints of the weight box.
    maximum : float, or array-like,
              numeric or named list defining the maximum constraints of the weight box.
    kind :    str,
              string of kind of constraint.    
    enabled : bool, default = True
              bool to enable or disable constraints.
    message : bool, default = True
              bool to enable or disable messages.
    kwargs :  additional key word arguments, optional
              any additional constraint argument to be passed.
    
    Returns
    -------
    Add box constraints to object portfolio_spec of the specified input. 
    
    See Also
    --------
    add_constraints
    box_constraints
    group_constraints
    weight_sum_constraint
    turnover_constraint
    diversification_constraint
    position_limit_constraint
    return_constraint
    factor_exposure_constraint
    transaction_cost_constraint
    leverage_exposure_constraint


    Examples
    --------
    >>> add_constraint(portfolio, kind = 'box', minimum = [0.9, -0.5,-0.5, 0.1], maximum = 1)
"""
    import numpy as np
    import pandas as pd

    import math as math
    import random



    
    nassets = len(assets)
    if kind == 'long_only':
        minimum = np.repeat(0, nassets)
        maximum = np.repeat(1, nassets)
    if type(minimum) == list and type(maximum) == list:
        if len(minimum) > 1 and len(maximum) > 1:
            if len(minimum) != len(maximum):
                raise ValueError('len(minimum) and len(maximum) must be same')
            if len(minimum) == nassets and len(maximum) == nassets:
                minimum = minimum
                maximum = maximum
    if type(minimum) == int or type(minimum) == float:
        minimum = np.repeat(minimum, nassets)
    if type(maximum) == int or type(maximum) == float:
        maximum = np.repeat(maximum, nassets)
    names = ['minimum', 'maximum', 'enabled']
    return(dict(zip(names, [minimum, maximum,enabled])))


# In[6]:


def group_constraint(assets, groups, group_min, group_max, kind = 'group', enabled = True, message = False):
    """
    
    
    Group constraints determine the grouping of assets, group weights,
    and the number of positions of the groups (i.e. non-zero weights).
    
    Parameters
    ----------  
    assets : Int, or array-like,
            Number of assets or, as an alternative, a named asset list specifying initial weights.
    groups : Dict, 
            dictionary specifying the assets groups.
    group_min : float, or array-like,
            numeric or named list defining the minimum constraints of the weight group. 
    group_max : float, or array-like,
            numeric or named list defining the maximum constraints of the weight group.
    kind : str,
            string of kind of constraint.    
    enabled : bool, default = True
            bool to enable or disable constraints.
    message : bool, default = False
            bool to enable or disable messages.
    
    Returns
    -------
    Adds group constraints to object portfolio_spec of the specified input. 
    
    See Also
    --------
    add_constraints
    box_constraints
    group_constraints
    weight_sum_constraint
    turnover_constraint
    diversification_constraint
    position_limit_constraint
    return_constraint
    factor_exposure_constraint
    transaction_cost_constraint
    leverage_exposure_constraint

   
    Examples
    --------
    >>> add_constraint(portfolio, kind = 'group', group_min = [0.9, -0.5,-0.5, 0.1], 
        group_max =1, groups = {'eqity':[0,3], 'debt':[1,2]})

"""
    import numpy as np
    import pandas as pd
    import math as math
    import random




    

    if type(groups) != dict:
        raise TypeError('Groups must be passed as a pandas dict')
    if type(assets) == int:
        nassets = assets
    else:
        nassets = len(assets)
    
    group_names = list(groups.keys())
    ngroups = len(group_names)
    if type(group_min) == int or type(group_min) == float:
        group_min = np.repeat(group_min, ngroups)
        if len(group_min) != ngroups:
            raise ValueError('len(group_min) must equal ngroups')
    if type(group_max) == int or type(group_max) == float:
        group_max = np.repeat(group_max, ngroups)
        if len(group_max) != ngroups:
            raise ValueError('len(group_max) must equal ngroups')
    if group_names != None:
        group_label = group_names
    if group_names == None and group_label == None:
        group_label = [0]*ngroups
        for i in range(0, ngroups):
            group_label[i] = ''.join(['group', str(i)])
    if len(group_label) != ngroups:
        raise ValueError('group_labels must match ngroups')
    names = ['groups', 'group_min', 'group_max', 'group_label', 'enabled']
    return(dict(zip(names, [groups, group_min, group_max, group_label, enabled])))


# In[7]:


def weight_sum_constraint(assets, kind = 'weight_sum', min_sum = 0.99, max_sum = 1.01, enabled = True, message = True):
    """


    The constraint determines the sum of the weights of the upper and lower limits.
    This function is called add.constraint when the type is defined as "weight sum", 
    "leverage", "full investment", "dollar neutral" or "active"
    
    Parameters
    ----------
    assets : Int, or array-like,
            Number of assets or, as an alternative, a named asset list specifying initial weights.
    kind : str,
            character kind of the constraint.
    min_sum : float,
            Minimum sum of all weights of assets, default 0.99
    max_sum : float, 
            Maximum sum of all weights of assets, default 1.01    
    enabled : bool, default = True
            bool to enable or disable constraints.
    message : bool, default = True
            bool to enable or disable messages.  

    
    Returns
    -------
    Adds weight_sum constraint to object portfolio_spec of the specified input. 
    
    See Also
    --------
    add_constraints
    box_constraints
    group_constraints
    weight_sum_constraint
    turnover_constraint
    diversification_constraint
    position_limit_constraint
    return_constraint
    factor_exposure_constraint
    transaction_cost_constraint
    leverage_exposure_constraint


    Examples
    --------

    >>> # adding weight_sum constraint
    >>> add_constraint(portfolio, kind = 'weight_sum', min_sum = 0.9, max_sum = .95)
    >>> # special case of weight_sum is dollar_neutral/active or full_investment
    >>> add_constraint(portfolio, kind = 'dollar_neutral')
    >>> add_constraint(portfolio, kind = 'full_investment')

"""
    import numpy as np
    import pandas as pd
    import math as math
    import random






    
    if kind == 'full_investment':
        min_sum = 1
        max_sum = 1
    if kind == 'dollar_neutral':
        min_sum = 0
        max_sum = 0
    if kind == 'active':
        min_sum = 0
        max_sum = 0
    names = ['min_sum', 'max_sum', 'enabled']
    return(dict(zip(names, [min_sum, max_sum,enabled])))


# In[8]:


def turnover_constraint(assets, turnover_target,kind = 'turnover',  enabled = True, message = True):
    """
    Target turnover value is determined under the turnover constraint. 
    When type=”turnover” is stated, the function is called by add.constraint
   
    Parameters
    ----------
    asset : Int, or array-like,
            Number of assets or, as an alternative, a named asset list specifying initial weights.
    kind :  str,
            character kind of the constraint.    
    enabled : bool, default = True
            bool to enable or disable constraints.
    message : bool, default = True
            bool to enable or disable messages.
    
    Returns
    -------
    Adds  turnover constraints to object portfolio_spec of the specified input. 
    
    See Also
    --------
    add_constraints
    box_constraints
    group_constraints
    weight_sum_constraint
    turnover_constraint
    diversification_constraint
    position_limit_constraint
    return_constraint
    factor_exposure_constraint
    transaction_cost_constraint
    leverage_exposure_constraint

       
    Examples
    --------
>>> #turnover constraint
>>> add_constraint(portfolio, kind = 'turnover', turnover_target = 0.1)

""" 
    return(dict(zip(['turnover_target', 'enabled'],[turnover_target, enabled])))


# In[9]:


def diversification_constraint(assets, div_target, kind = 'diversification', enabled = True, message = True):
    """
    Target diversification value is specified under the diversification constraint.
    
    This function is called by add.constraint when type=”diversification” is mentioned.
    
    Parameters
    ----------
    assets : Int, or array-like,
            Number of assets or, as an alternative, a named asset list specifying initial weights.
    div_target : float,
            diversification target value
    kind : str,
            string of the kind of constraint.    
    enabled : bool, default = True
            bool to enable or disable constraints.
    message : bool, default = True
            bool to enable or disable messages.

    
    Returns
    -------
    Adds diversification constraints to object portfolio_spec of the specified input. 
    
    See Also
    --------
    add_constraints
    box_constraints
    group_constraints
    weight_sum_constraint
    turnover_constraint
    diversification_constraint
    position_limit_constraint
    return_constraint
    factor_exposure_constraint
    transaction_cost_constraint
    leverage_exposure_constraint

       
    Examples
    --------
    >>> #diversification constraint for diversification target in a portfolio
    >>> add_constraint(portfolio, kind = 'diversification', div_target = 0.1)
"""

    import numpy as np
    import random
    import pandas as pd
    import math as math
    import random



    
    return(dict(zip(['diversification_target', 'enabled'], [div_target,enabled])))


# In[10]:


def position_limit_constraint(assets,  max_pos = None,kind = 'position', max_pos_long = None, 
                              max_pos_short = None, enabled = True, message = True):
    """

    When type=”position_limit” is mentioned, this function is called by add.constraint. 

    The maximum number of positions as well as the maximum number of long and short positions
    are determined by the user using this function.
    
    Parameters
    ----------
    assets : int, or array-like,
            named list of assets determining initial weights.
    max_pos : int,
            maximum number of assets with non-zero weights.
    kind : str,
            string of kind of constraint
    max_pos_long : Int, optional
            maximum number of assets with long positions.
    max_pos_short : Int, optional
            maximum number of assets with short positions.
    enabled : bool, default = True
            bool to enable or disable constraints.
    message : bool, default = True
            bool to enable or disable messages.
   
    
    Returns
    -------
    Adds position_limit constraints to object portfolio_spec of the specified input. 
    
    See Also
    --------
    add_constraints
    box_constraints
    group_constraints
    weight_sum_constraint
    turnover_constraint
    diversification_constraint
    position_limit_constraint
    return_constraint
    factor_exposure_constraint
    transaction_cost_constraint
    leverage_exposure_constraint

        
    Examples
    --------
>>> #position_limit is a constraint to restrict max position and also max long/short positions
>>> add_constraint(portfolio, kind = 'position_limit', max_pos = 3, max_pos_long = 2, max_pos_short = 2)

"""

    import numpy as np
    import random
    import pandas as pd
    import math as math


    
    nassets = len(assets)
    if type(max_pos) != int:
        raise TypeError('max_pos must be of type int')
    if max_pos < 0:
        raise ValueError('max_pos must a positive number')
    if max_pos > nassets:
        raise ValueError('max_pos must be less than or equal to number of assets')
        max_pos = nassets
    max_pos = int(max_pos)
    if type(max_pos_long) != int:
        raise TypeError('max_pos_long must be of type int')
    if max_pos_long < 0:
        raise ValueError('max_pos_long must a positive number')
    if max_pos_long > nassets:
        raise ValueError('max_pos_long must be less than or equal to number of assets')
        max_pos_long = nassets
    max_pos_long = int(max_pos_long)
    if type(max_pos_short) != int:
        raise TypeError('max_pos_short must be of type int')
    if max_pos_short < 0:
        raise ValueError('max_pos_short must a positive number')
    if max_pos_short > nassets:
        raise ValueError('max_pos_short must be less than or equal to number of assets')
        max_pos_short = nassets
    max_pos_short = int(max_pos_short)
    names = ['max_pos', 'max_pos_long', 'max_pos_short','enabled']
    return(dict(zip(names, [max_pos, max_pos_long, max_pos_short,enabled])))


# In[11]:


def return_constraint(assets,  return_target, kind = 'return',enabled = True, message = False):
    """

    Target mean return value is determined by the return constraint. 
    When type=”return” is mentioned, this function is called by add_constraint.
    
    Parameters
    ----------
    assets :. int, or array-like,
            named list of assets determining initial weights.
    kind : str,
            string of kind of constraints.    
    enabled : bool, default = True
            bool to enable or disable constraints.
    messages : bool, default = False
            bool to enable or disable messages.
  
    Returns
    -------
    Adds return constraints to object portfolio_spec of the specified input. 
    
    See Also
    --------
    
    add_constraints
    box_constraints
    group_constraints
    weight_sum_constraint
    turnover_constraint
    diversification_constraint
    position_limit_constraint
    return_constraint
    factor_exposure_constraint
    transaction_cost_constraint
    leverage_exposure_constraint

  
    Examples
    --------
    >>> #return constraint to add a target mean historical return to the portfolio
    >>> add_constraint(portfolio, kind = 'return', return_target = 0.0018)


"""



    
    return(dict(zip(['return_target','enabled'], [return_target,enabled])))


# In[12]:


def factor_exposure_constraint(assets, B, lower, upper, kind = 'factor_exposure', enabled = True, message = False):
    """
    Add a factor exposure constraint for "K" different factos in a portdolio and also their exposure to 
    particular asset.

    
    Parameters
    ----------
    assets : int, or array-like,
            named list of assets specifying initial weights.
    B : matrix-like,
        matrix or list of risk factor exposures
    lower : int, or array-like
        list of lower limits of risk factor exposure constraints.
    upper : int, or array-like
        list of upper limits of risk factor exposure constraints.
    kind : str,
        string of kind of constraints.    
    enabled : bool, default = True
        bool to enable or disable constraints.
    message : bool, default = False
        bool to enable or disable messages.
   
    
    Returns
    -------
    Adds factor_exposure constraints to object portfolio_spec of the specified input. 
    
    See Also
    --------
    add_constraints
    box_constraints
    group_constraints
    weight_sum_constraint
    turnover_constraint
    diversification_constraint
    position_limit_constraint
    return_constraint
    factor_exposure_constraint
    transaction_cost_constraint
    leverage_exposure_constraint

    Examples
    --------
    >>> #Factor_exposure constraint is used to test portfolio with their impact on certaint factors
    >>> # factors can be single or multiple with lower and upper limit. 
    >>> # B can be a N*K matrix for N assets and K factors.
    >>> # lower and upper arguments can be float for single factor and list for multiple factors
    >>> add_constraint(portfolio, kind = 'factor_exposure', 
                  B = np.matrix([[1.2,1.3],
                                 [2.3,1.4],
                                 [1.4,0.9],
                                 [3.4,1.2]]),
                   lower = [0.8,1.4], upper = [2,2.4])

"""

    import numpy as np
    import pandas as pd

    import math as math
    import random




    nassets = len(assets)
    ##Factor Exposure Constraint
    if type(B) == list or type(B) == np.ndarray:
        if len(B) != nassets:
            raise ValueError('length of B must equal nassets')
        if type(lower) != float:
            raise TypeError('lower must be a float')
        if type(upper) != float:
            raise TypeError('upper must be a float')
        B = np.transpose(np.matrix(B))
        if type(B) == np.matrix:
            if B.shape[0] != nassets:
                raise ShapeError('number of rows must equal nassets')
            if B.shape[1] != 1:
                raise ShapeError('number of columns must equal lower')
    names = ['B', 'lower', 'upper', 'enabled']
    return(dict(zip(names, [B, lower, upper, enabled])))


# In[13]:


def transaction_cost_constraint(assets, ptc, kind = 'transaction', enabled = True, message = True):
    """
    A proportional cost value is determined under the transaction cost constraint.
    When type=”transaction_cost” is specified, this function is called by add.constraint.
    
    Parameters
    ----------
    assets : int, or array-like,
            number of assets, or optionally a named list of assets determining initial weights.
    ptc : float,
            proportional transaction cost value.  
    kind : str,
            string of kind of constraint 
    enabled : bool, default = True
            bool to enable or disable constraints.
    message : bool, default = True
            bool to enable or disable messages.
   
    
    Returns
    -------
    Adds transaction _cost constraints to object portfolio_spec of the specified input. 
    
    See Also
    --------
    add_constraints
    box_constraints
    group_constraints
    weight_sum_constraint
    turnover_constraint
    diversification_constraint
    position_limit_constraint
    return_constraint
    factor_exposure_constraint
    transaction_cost_constraint
    leverage_exposure_constraint

    
    Examples
    --------
    >>> #adds a tranaction cost constraint on portfolio
    >>> add_constraint(portfolio, kind = 'transaction_cost', ptc = 0.1)

"""

    


    return(dict(zip(['ptc','enabled'], [ptc,enabled])))


# In[14]:


def leverage_exposure_constraint(assets, leverage, kind = 'leverage_exposure', enabled = True, message = True):
    
    """
    A maximum leverage where leverage is stated as the total of the absolute value of the weights
    is determined under the leverage_exposure constraint.    
    
    Parameters
    ----------
    assets : int, or array-like,
            leverage : maximum leverage value
    kind : str,
            string of kind of constraints    
    enabled : bool, default = True
            bool to enable or disable constraints.
    message : bool, default = True
            bool to enable or disable messages.
   
    
    Returns
    -------
    Adds leverage_exposure constraints to object portfolio_spec of the specified input. 
    
    See Also
    --------
    add_constraints
    box_constraints
    group_constraints
    weight_sum_constraint
    turnover_constraint
    diversification_constraint
    position_limit_constraint
    return_constraint
    factor_exposure_constraint
    transaction_cost_constraint
    leverage_exposure_constraint

    
    Examples
    --------
    >>> #constraint on the leverage of portfolio
    >>> add_constraint(portfolio, kind = 'leverage_exposure', leverage = 0.8)

"""


    
    return(dict(zip(['leverage','enabled'], [leverage,enabled])))


# In[15]:


def diversification(weights):
    """
    Diversification is stated as 1 minus the total of the squared weights

    Diversification function to compute as a constraint
    
    Parameters
    ----------
    weights : array-like,
            list of weights of assets. 
   
    Returns
    -------
    returns diversification given the weights
    
    See Also
    --------
    turnover
    
    
    Examples
    --------
    >>> w = [0.2,0.3,0.5]
    >>> turnover(w)
"""

    
    div = 1- sum(weights**2)
    return(div)


# In[16]:


def get_constraints(portfolio):
    """
    Helper functionality to get the constraints allowed out of the portfolio object 
    
    Parameters
    ----------
    portfolio : portfolio_spec,
                an object of class portfolio_spec. 
  
    Returns
    -------
    returns dictionary of constraints extracted from the portfolio_spec object
    
    Examples
    --------
    >>> port = portfolio_spec(3)
    >>> add_constraint('dollar_neutral')
    >>> get_constraint(port)
"""

    return(portfolio.constraints)


# In[17]:


def add_objective(portfolio, kind, name, arguments = None, constraints = None, enabled = True, message = True, **kwargs):

    """

    This function is the primary function of adding and updating business goals in a portfolio.spec type object.
    General interface, including risk, return, and risk budget, to add optimization goals.   

    Parameters
    ----------
    portfolio : portfolio_spec,
            an object of class portfolio_spec. 
    kind : str,
        the character form of the goal to be added or changed, currently 'return',' risk',
        ‘risk_budget',' quadratic_utility’, or ‘weight_concentration’. 
    name : str,
        The target name should correspond to a feature, although we will attempt to make allowances
        arguments : Default arguments to be transferred when executed on an objective function.
    enabled : bool, default = True
        bool to enable or disable constraints.
    message : bool, default = True
        bool to enable or disable messages.
    kwargs : additional key word arguments, optional
        any additional constraint argument to be passed.

    
    Returns
    -------
    Adds objective to object portfolio_spec of the specified input. 
    
    See Also
    --------
    add_objectives
    portfolio_risk_objectives
    risk_budget_objective
    turnover_objective
    minmax_objective
    weight_constraint_objective
 
    Notes
    -----
    In general, you will define your objective as one of the following types:
    ’return’, ’risk’, ’risk_budget’, or ’weight_concentration’.
    
    These have special handling and intelligent defaults for dealing with the function
    most likely to be used as objectives, including mean, median, VaR, ES, etc.
    Objectives of type ’turnover’ and ’minmax’ are also supported.
    
    Examples
    --------
    >>> port = portfolio_spec(assets = 5)
    >>> import pandas_datareader as pdr
    >>> aapl = pdr.get_data_yahoo('AAPL')
    >>> msft = pdr.get_data_yahoo('MSFT')
    >>> tsla = pdr.get_data_yahoo('TSLA')
    >>> uber = pdr.get_data_yahoo('UBER')
    >>> amzn = pdr.get_data_yahoo('AMZN')
    >>> port = pd.DataFrame({'aapl': pd.DataFrame.reset_index(aapl).iloc[:,6], 'msft':pd.DataFrame.reset_index(msft).iloc[:,6],
        'tsla': pd.DataFrame.reset_index(tsla).iloc[:,6], 'uber':pd.DataFrame.reset_index(uber).iloc[:,6],
        'amzn': pd.DataFrame.reset_index(amzn).iloc[:,6]})
    >>> port_ret = port.pct_change().dropna()
    >>> R = port_ret
    >>> add_constraint('long_only')
    >>> add_constraint('full_investment')
    >>> #adding objectives
    >>> add_objective(kind = 'return', name = 'mean', target = 0.0018)
    >>> add_objective(kind = 'portfolio_risk', name = 'std', target = 0.015)
    >>> add_objective(kind = 'risk_budget', name = 'risk_budget')
    >>> add_objective(kind = 'weight_conc', name = 'HHI', target = 0.11)
    >>> add_objective(kind = 'performance_metrics', name = 'sharpe', target = 0.13)
    >>> # add a custom objective by first defining it.
    >>> def sortino_ratio(w,R):
            #SOME CODE
    >>> add_objective(kind = 'performance_metrics', name = {'sortino':sortino_ratio}, target = 0.35)
    NOTE: The output of sortino_ratio or other custom function in objective must be a float.
    NOTE: you can also add other custom function in other kind of objective in similar methd.


"""

    import numpy as np
    import pandas as pd

    import math as math
    import random



    
    if kind == None:
        raise ValueError('You must specify thekind')
    if arguments == None:
        arguments = dict()
    if type(arguments) != dict:
        raise TypeError('You must specify arguments in a type dict')
    if name == None:
        raise ValueError('You must enter the name of the objective')
    assets = portfolio.assets
    tmp_objective = None
    if kind == 'return':
        tmp_objective = return_objective(name = name,  enabled = enabled, arguments = arguments, **kwargs)
    elif kind == 'risk' or kind == 'portfolio_risk':
        tmp_objective = portfolio_risk_objective(name = name,  enabled = enabled, arguments = arguments, **kwargs)
    elif kind == 'risk_budget':
        tmp_objective = risk_budget_objective(name = name, assets = assets, enabled = enabled, arguments = arguments, **kwargs)
    elif kind == 'turnover':
        tmp_objective = turnover_objective(name = name, enabled = enabled, arguments = arguments, **kwargs)
    elif kind == 'minmax':
        tmp_objective = minmax_objective(name = name, enabled = enabled, arguments = arguments, **kwargs)
    elif kind == 'weight_conc' or kind == 'weight_concentration':
        tmp_objective = weight_concentration_objective(name = name, enabled = enabled, arguments = arguments, **kwargs)
    elif kind == 'performance_metrics':
        tmp_objective = performance_metrics_objective(name = name, enabled = enabled, arguments = arguments, **kwargs)
    else:
        return(portfolio)
    if portfolio.objectives == None:
        portfolio.objectives = dict()
        portfolio.objectives[kind] = tmp_objective
    else:
        portfolio.objectives[kind] = tmp_objective
    return(portfolio)


# In[18]:


def return_objective(name, target, arguments, multiplier = -1, enabled = True):
    """
    
    We'll try minimizing the risk metric if the target is null.    
    Constructor for the portfolio_risk_objective class.
    
    Parameters
    ----------
    name : str,
        The target name should correspond to a feature, although we will attempt to make allowances
    target : float,
        Univariate goal for the target.
    arguments : dict, optional
        Default arguments to be transferred when executed on an objective function.
    multiplier : int, optional
        Multiplier to be added to the target, typically 1 or -1.
    enabled : bool, default = True
        bool to enable or disable constraints.

    Returns
    -------
    Adds return_objective to object portfolio_spec of the specified input. 
    
    See Also
    --------
    add_objectives
    portfolio_risk_objectives
    risk_budget_objective
    turnover_objective
    minmax_objective
    weight_constraint_objective
   
    
    Examples
    --------
    >>> port = portfolio_spec(assets = 5)
    >>> import pandas_datareader as pdr
    >>> aapl = pdr.get_data_yahoo('AAPL')
    >>> msft = pdr.get_data_yahoo('MSFT')
    >>> tsla = pdr.get_data_yahoo('TSLA')
    >>> uber = pdr.get_data_yahoo('UBER')
    >>> amzn = pdr.get_data_yahoo('AMZN')
    >>> port = pd.DataFrame({'aapl': pd.DataFrame.reset_index(aapl).iloc[:,6], 'msft':pd.DataFrame.reset_index(msft).iloc[:,6],
        'tsla': pd.DataFrame.reset_index(tsla).iloc[:,6], 'uber':pd.DataFrame.reset_index(uber).iloc[:,6],
        'amzn': pd.DataFrame.reset_index(amzn).iloc[:,6]})
    >>> port_ret = port.pct_change().dropna()
    >>> R = port_ret
    >>> add_constraint('long_only')
    >>> add_constraint('full_investment')
    >>> #adding objectives
    >>> add_objective(kind = 'return', name = 'mean', target = 0.0018)
    >>> # add a custom objective by first defining it.
    >>> def sortino_ratio(w,R):
            #SOME CODE
    >>> add_objective(kind = 'performance_metrics', name = {'sortino':sortino_ratio}, target = 0.35)
    NOTE: The output of sortino_ratio or other custom function in objective must be a float.
    NOTE: you can also add other custom function in other kind of objective in similar methd.




"""
    import numpy as np
    import pandas as pd
    import math as math
    import random





    return(dict(zip(['target', 'multiplier', 'arguments','enabled','name'], [target, multiplier, arguments,enabled,name])))


# In[19]:


def portfolio_risk_objective(name, target, arguments, multiplier = -1, enabled = True):
    """
    We'll try minimizing the risk metric if the target is null.    
    Constructor for the portfolio_risk_objective class.
    
    Parameters
    ----------
    name : str,
        The target name should correspond to a feature, although we will attempt to make allowances
    target : float,
        Univariate goal for the target.
    arguments : dict, optional
        Default arguments to be transferred when executed on an objective function.
    multiplier : int, optional
        Multiplier to be added to the target, typically 1 or -1.
    enabled : bool, default = True
        bool to enable or disable constraints.


    Returns
    -------
    Adds portfolio risk objectives to object portfolio_spec of the specified input. 
    
    See Also
    --------
    
    add_objectives
    portfolio_risk_objectives
    risk_budget_objective
    turnover_objective
    minmax_objective
    weight_constraint_objective
   
    
    Examples
    --------
    >>> port = portfolio_spec(assets = 5)
    >>> import pandas_datareader as pdr
    >>> aapl = pdr.get_data_yahoo('AAPL')
    >>> msft = pdr.get_data_yahoo('MSFT')
    >>> tsla = pdr.get_data_yahoo('TSLA')
    >>> uber = pdr.get_data_yahoo('UBER')
    >>> amzn = pdr.get_data_yahoo('AMZN')
    >>> port = pd.DataFrame({'aapl': pd.DataFrame.reset_index(aapl).iloc[:,6], 'msft':pd.DataFrame.reset_index(msft).iloc[:,6],
        'tsla': pd.DataFrame.reset_index(tsla).iloc[:,6], 'uber':pd.DataFrame.reset_index(uber).iloc[:,6],
        'amzn': pd.DataFrame.reset_index(amzn).iloc[:,6]})
    >>> port_ret = port.pct_change().dropna()
    >>> R = port_ret
    >>> add_constraint('long_only')
    >>> add_constraint('full_investment')
    >>> #adding objectives
    >>> add_objective(kind = 'portfolio_risk', name = 'std', target = 0.015)
    >>> # add a custom objective by first defining it.
    >>> def sortino_ratio(w,R):
            #SOME CODE
    >>> add_objective(kind = 'performance_metrics', name = {'sortino':sortino_ratio}, target = 0.35)
    NOTE: The output of sortino_ratio or other custom function in objective must be a float.
    NOTE: you can also add other custom function in other kind of objective in similar methd.

   

"""
    import numpy as np
    import pandas as pd
    import math as math
    import random




    return(dict(zip(['target', 'multiplier', 'arguments','enabled','name'], [target, multiplier, arguments,enabled,name])))


# In[20]:


def risk_budget_objective(assets, name, min_prisk = None, max_prisk = None, target = None, arguments = None, multiplier = 1, enabled = True, min_concentration = False, min_difference = False):

    """
    
    Constructor for the risk_budget_objective.
    
    Parameters
    ----------
    assets : int, or array-like,
        The asset list to be used should come from object constraints.
    name : str,
        The target name should correspond to a feature, although we will attempt to make allowances.
    min_prisk : float,
        minimum percentage risk contribution
    max_prisk : float,
        maximum percentage risk contribution
    arguments : dict, optional
        Default arguments to be transferred when executed on an objective function.
    multiplier : int, optional
        Multiplier to be added to the target, typically 1 or -1.
   
    
    Returns
    -------
    Adds risk budget objective to object portfolio_spec of the specified input. 
    
    See Also
    --------
    add_objectives
    portfolio_risk_objectives
    risk_budget_objective
    turnover_objective
    minmax_objective
    weight_constraint_objective
    
   Examples
    --------
    >>> port = portfolio_spec(assets = 5)
    >>> import pandas_datareader as pdr
    >>> aapl = pdr.get_data_yahoo('AAPL')
    >>> msft = pdr.get_data_yahoo('MSFT')
    >>> tsla = pdr.get_data_yahoo('TSLA')
    >>> uber = pdr.get_data_yahoo('UBER')
    >>> amzn = pdr.get_data_yahoo('AMZN')
    >>> port = pd.DataFrame({'aapl': pd.DataFrame.reset_index(aapl).iloc[:,6], 'msft':pd.DataFrame.reset_index(msft).iloc[:,6],
        'tsla': pd.DataFrame.reset_index(tsla).iloc[:,6], 'uber':pd.DataFrame.reset_index(uber).iloc[:,6],
        'amzn': pd.DataFrame.reset_index(amzn).iloc[:,6]})
    >>> port_ret = port.pct_change().dropna()
    >>> R = port_ret
    >>> add_constraint('long_only')
    >>> add_constraint('full_investment')
    >>> #adding objectives
    >>> add_objective(kind = 'risk_budget', name = 'risk_budget')
    >>> add_objective(kind = 'weight_conc', name = 'HHI', target = 0.11)
    >>> add_objective(kind = 'performance_metrics', name = 'sharpe', target = 0.13)
    >>> # add a custom objective by first defining it.
    >>> def sortino_ratio(w,R):
            #SOME CODE
    >>> add_objective(kind = 'performance_metrics', name = {'sortino':sortino_ratio}, target = 0.35)
    NOTE: The output of sortino_ratio or other custom function in objective must be a float.
    NOTE: you can also add other custom function in other kind of objective in similar methd.




"""
    import numpy as np
    import pandas as pd
    import math as math
    import random




    nassets = len(assets)
    if arguments == None:
        arguments = dict()
        arguments['portfolio_method'] = 'component'
    if type(min_prisk) == list and type(max_prisk) == list:
        if len(min_prisk) > 1 and len(max_prisk) > 1:
            if len(min_prisk) != len(max_prisk):
                raise ValueError('len(min_prisk) must equal len(max_prisk)')
    if type(min_prisk) == int:
        min_prisk = np.repeat(min_prisk, nassets)
    if type(max_prisk) == int:
        max_prisk = np.repeat(max_prisk, nassets)
    if min_prisk is None and max_prisk is None:
        min_concentration = True
    names = ['target', 'arguments', 'multiplier', 'min_prisk', 'max_prisk', 'min_concentration', 'min_difference','enabled','name']
    obj = [target, arguments, multiplier, min_prisk, max_prisk, min_concentration, min_difference,enabled,name]
    return(dict(zip(names, obj)))


# In[21]:


def turnover_objective(name, arguments, target = None, multiplier = 1, enabled = True):

    """
    We'll try minimizing the turnover metric if the goal is null.    
    Constructor for the turnover_objective class.
    
    Parameters
    ----------
    name : str,
        The target name should correspond to a feature, although we will attempt to make allowances.
    arguments : dict,
        Default arguments to be transferred when executed on an objective function.
    target : float,
        Univariate goal for the target.
    multiplier : int, optional
        Multiplier to be added to the target, typically 1 or -1.  
    enabled : bool, default = True
        bool to enable or disable constraints.
    
   
    
    Returns
    -------
    Adds turnover_objective to object portfolio_spec of the specified input. 
    
    See Also
    --------
    add_objectives
    portfolio_risk_objectives
    risk_budget_objective
    turnover_objective
    minmax_objective
    weight_constraint_objective

    
    Examples
    --------
    >>> port = portfolio_spec(assets = 5)
    >>> import pandas_datareader as pdr
    >>> aapl = pdr.get_data_yahoo('AAPL')
    >>> msft = pdr.get_data_yahoo('MSFT')
    >>> tsla = pdr.get_data_yahoo('TSLA')
    >>> uber = pdr.get_data_yahoo('UBER')
    >>> amzn = pdr.get_data_yahoo('AMZN')
    >>> port = pd.DataFrame({'aapl': pd.DataFrame.reset_index(aapl).iloc[:,6], 'msft':pd.DataFrame.reset_index(msft).iloc[:,6],
        'tsla': pd.DataFrame.reset_index(tsla).iloc[:,6], 'uber':pd.DataFrame.reset_index(uber).iloc[:,6],
        'amzn': pd.DataFrame.reset_index(amzn).iloc[:,6]})
    >>> port_ret = port.pct_change().dropna()
    >>> R = port_ret
    >>> add_constraint('long_only')
    >>> add_constraint('full_investment')
    >>> #adding objectives
    >>> 
    >>> add_objective(kind = 'turnover', name = 'turnover', target = 0.2)
    >>> add_objective(kind = 'performance_metrics', name = 'sharpe', target = 0.13)
    >>> # add a custom objective by first defining it.
    >>> def sortino_ratio(w,R):
            #SOME CODE
    >>> add_objective(kind = 'performance_metrics', name = {'sortino':sortino_ratio}, target = 0.35)
    NOTE: The output of sortino_ratio or other custom function in objective must be a float.
    NOTE: you can also add other custom function in other kind of objective in similar methd.



"""


    return(dict(zip(['target', 'multiplier', 'arguments','enabled','name'], [target, multiplier, arguments,enabled,name])))


# In[22]:


def minmax_objective(name, minimum, maximum, multiplier = 1, arguments = None, target = None, enabled = True):

    """
    
    This target allows to determine min and max goals.
    Constructor for the tmp_minmax_objective.
    
    Parameters
    ----------
    name : str,
        The target name should correspond to a feature, although we will attempt to make allowances.
    minimum : float,
        minimum value
    maximum : float,
        maximum value
    multiplier : int, optional
        Multiplier to be added to the target, typically 1 or -1.
    arguments : dict,
        Default arguments to be transferred when executed on an objective function.
    target : float,
        Univariate goal for the target.
    enabled : bool, default = True
        bool to enable or disable constraints.
   
    
    Returns
    -------
    Adds minmax_objective to object portfolio_spec of the specified input. 
    
    See Also
    --------
    add_objectives
    portfolio_risk_objectives
    risk_budget_objective
    turnover_objective
    minmax_objective
    weight_constraint_objective


    Examples
    --------
    >>> port = portfolio_spec(assets = 5)
    >>> import pandas_datareader as pdr
    >>> aapl = pdr.get_data_yahoo('AAPL')
    >>> msft = pdr.get_data_yahoo('MSFT')
    >>> tsla = pdr.get_data_yahoo('TSLA')
    >>> uber = pdr.get_data_yahoo('UBER')
    >>> amzn = pdr.get_data_yahoo('AMZN')
    >>> port = pd.DataFrame({'aapl': pd.DataFrame.reset_index(aapl).iloc[:,6], 'msft':pd.DataFrame.reset_index(msft).iloc[:,6],
        'tsla': pd.DataFrame.reset_index(tsla).iloc[:,6], 'uber':pd.DataFrame.reset_index(uber).iloc[:,6],
        'amzn': pd.DataFrame.reset_index(amzn).iloc[:,6]})
    >>> port_ret = port.pct_change().dropna()
    >>> R = port_ret
    >>> add_constraint('long_only')
    >>> add_constraint('full_investment')
    >>> #adding objectives
    >>> add_objective(kind = 'risk', name = 'std', target = 0.015)
    NOTE: you can add other custom function in other kind of objective in similar methd. See add_objective


"""
    import numpy as np
    import pandas as pd

    import math as math
    import random




    names = ['minimum', 'maximum', 'multiplier', 'arguments', 'target','enabled','name']
    return(dict(zip(names, [minimum, maximum, multiplier, arguments, target,enabled,name])))


# In[23]:


def weight_concentration_objective(name, conc_aversion, conc_groups = None, multiplier = 1,arguments = None, enabled = True):
    
    """    
    Using the HHI as a concentration scale, this feature penalizes weight concentration
    Constructor for objective of weight concentration.
    
    Parameters
    ----------
    name : str,
        The name of the concentration measure is currently only supported by "HHI".
    conc_aversion : float,
        concentration value(s) of aversion
    conc_groups : dict,
        A dictionary defining the asset classes. Similar to 'group constraint' groups.
    multiplier : int, optional
        Multiplier to be added to the target, typically 1 or -1.
    arguments : dict, optional
        Default arguments to be transferred when executed on an objective function.
    enabled : bool, default = True
        bool to enable or disable constraints.
   
    
    Returns
    -------
    Adds weight constraints objective to object portfolio_spec of the specified input. 
    
    See Also
    --------
    add_objectives
    portfolio_risk_objectives
    risk_budget_objective
    turnover_objective
    minmax_objective
    weight_constraint_objective
    
    Examples
    --------
    >>> port = portfolio_spec(assets = 5)
    >>> import pandas_datareader as pdr
    >>> aapl = pdr.get_data_yahoo('AAPL')
    >>> msft = pdr.get_data_yahoo('MSFT')
    >>> tsla = pdr.get_data_yahoo('TSLA')
    >>> uber = pdr.get_data_yahoo('UBER')
    >>> amzn = pdr.get_data_yahoo('AMZN')
    >>> port = pd.DataFrame({'aapl': pd.DataFrame.reset_index(aapl).iloc[:,6], 'msft':pd.DataFrame.reset_index(msft).iloc[:,6],
        'tsla': pd.DataFrame.reset_index(tsla).iloc[:,6], 'uber':pd.DataFrame.reset_index(uber).iloc[:,6],
        'amzn': pd.DataFrame.reset_index(amzn).iloc[:,6]})
    >>> port_ret = port.pct_change().dropna()
    >>> R = port_ret
    >>> add_constraint('long_only')
    >>> add_constraint('full_investment')
    >>> add_objective(kind = 'weight_conc', name = 'HHI', target = 0.11)
    >>> add_objective(kind = 'performance_metrics', name = 'sharpe', target = 0.13)
    >>> # add a custom objective by first defining it.
    >>> def sortino_ratio(w,R):
            #SOME CODE
            
    >>> add_objective(kind = 'performance_metrics', name = {'sortino':sortino_ratio}, target = 0.35)
    NOTE: The output of sortino_ratio or other custom function in objective must be a float.
    NOTE: you can also add other custom function in other kind of objective in similar methd.



"""
    import numpy as np
    import pandas as pd
    import math as math
    import random





    if conc_groups != None:
        arguments['groups'] = conc_groups
        if type(conc_groups) != dict:
            raise TypeError('conc_groups must be a dictionary')
        if type(conc_aversion) == float:
            conc_aversion = np.repeat(conc_aversion, len(conc_groups))
        if len(conc_aversion) != len(conc_groups):
            raise ValueError('len(conc_aversion) must equal len(conc_groups)')
    elif conc_groups == None:
        if type(conc_aversion) != float:
            raise ValueError('conc_aversion must be type float when conc_groups is not defined')
    names = ['conc_aversion', 'conc_groups', 'arguments','multiplier','enabled','name']
    return(dict(zip(names, [conc_aversion, conc_groups, arguments,multiplier,enabled,name])))


# In[24]:


def performance_metrics_objective(name, arguments, target = None, multiplier = 1, enabled = True):
    """
    
    We'll try minimizing the performance_metric if the target is null.    
    Constructor for the portfolio_risk_objective class.
    
    Parameters
    ----------
    name : str,
        The target name should correspond to a feature, although we will attempt to make allowances
    target : float,
        Univariate goal for the target.
    arguments : dict, optional
        Default arguments to be transferred when executed on an objective function.
    multiplier : int, optional
        Multiplier to be added to the target, typically 1 or -1.
    enabled : bool, default = True
        bool to enable or disable constraints.

    Returns
    -------
    Adds portfolio risk objectives to object portfolio_spec of the specified input. 
    
    See Also
    --------
    add_objectives
    portfolio_risk_objectives
    risk_budget_objective
    turnover_objective
    minmax_objective
    weight_constraint_objective
   
    
    Examples
    --------
    >>> port = portfolio_spec(assets = 5)
    >>> import pandas_datareader as pdr
    >>> aapl = pdr.get_data_yahoo('AAPL')
    >>> msft = pdr.get_data_yahoo('MSFT')
    >>> tsla = pdr.get_data_yahoo('TSLA')
    >>> uber = pdr.get_data_yahoo('UBER')
    >>> amzn = pdr.get_data_yahoo('AMZN')
    >>> port = pd.DataFrame({'aapl': pd.DataFrame.reset_index(aapl).iloc[:,6], 'msft':pd.DataFrame.reset_index(msft).iloc[:,6],
        'tsla': pd.DataFrame.reset_index(tsla).iloc[:,6], 'uber':pd.DataFrame.reset_index(uber).iloc[:,6],
        'amzn': pd.DataFrame.reset_index(amzn).iloc[:,6]})
    >>> port_ret = port.pct_change().dropna()
    >>> R = port_ret
    >>> add_constraint('long_only')
    >>> add_constraint('full_investment')
    >>> #adding objectives
    >>> add_objective(kind = 'performance_metrics', name = 'sharpe', target = 0.13)
    >>> # add a custom objective by first defining it.
    >>> def sortino_ratio(w,R):
            #SOME CODE
            
    >>> add_objective(kind = 'performance_metrics', name = {'sortino':sortino_ratio}, target = 0.35)
    NOTE: The output of sortino_ratio or other custom function in objective must be a float.
    NOTE: you can also add other custom function in other kind of objective in similar methd.


"""


    return(dict(zip(['target', 'multiplier', 'arguments','enabled','name'], [target, multiplier, arguments,enabled,name])))


# In[25]:


def turnover(weights, wgt_init = None):
    """
    
    Turnover estimation of two weight lists.
    This is used as an objective function and is named when the user adds with add.objective 
    an objective of type turnover.
  
    Parameters
    ----------
    weights : array-like,
        Weights list from optimization.
    wgt_init : array-like, optional
        Initial weights list used for measuring turnover from.
    
    Returns
    -------
    returns the turnover of the portfolio given the weights
    
    See Also
    --------
    var_portfolio
    port_mean
        
    Examples
    --------
    >>> port = portfolio_spec(assets = 5)
    >>> import pandas_datareader as pdr
    >>> aapl = pdr.get_data_yahoo('AAPL')
    >>> msft = pdr.get_data_yahoo('MSFT')
    >>> tsla = pdr.get_data_yahoo('TSLA')
    >>> uber = pdr.get_data_yahoo('UBER')
    >>> amzn = pdr.get_data_yahoo('AMZN')
    >>> port = pd.DataFrame({'aapl': pd.DataFrame.reset_index(aapl).iloc[:,6], 'msft':pd.DataFrame.reset_index(msft).iloc[:,6],
        'tsla': pd.DataFrame.reset_index(tsla).iloc[:,6], 'uber':pd.DataFrame.reset_index(uber).iloc[:,6],
        'amzn': pd.DataFrame.reset_index(amzn).iloc[:,6]})
    >>> port_ret = port.pct_change().dropna()
    >>> R = port_ret
    >>> w = [0.2,0.3,0.4,-.4,0.5]
    >>> turnover(w)
"""
    import numpy as np
    import pandas as pd
    import math as math
    import random




    N = len(weights)
    if wgt_init == None:
        wgt_init = np.repeat(1/N, N)
    if len(wgt_init) != len(weights):
        raise ValueError('The length of weights and wgt_init must be the same')
    return(sum(abs(wgt_init-weights))/N)


# In[26]:


def var_portfolio(R, weights):

    """
    When var is an object for mean variance or quadratic utility optimization,
    this function is used to measure the portfolio variance through a call to constrained_objective.
    
    Main function to calculate portfolio variance
    
    Parameters
    ----------
    R : pd.DataFrame,
        return series
    weights: array-like,
        list of weights of assets.
    
    Returns
    -------
    returns the variance of the portfolio given the following weights.
    
    Notes
    -----
    
    \sigma_{p} = w \Sigma_{p} w^{T}
    
    See Also
    --------
    port_mean
    VaR
    cVaR
    
    
    Examples
    --------
    >>> # calculate the variance of portfolio
    >>> import pandas_datareader as pdr
    >>> aapl = pdr.get_data_yahoo('AAPL')
    >>> msft = pdr.get_data_yahoo('MSFT')
    >>> tsla = pdr.get_data_yahoo('TSLA')
    >>> uber = pdr.get_data_yahoo('UBER')
    >>> amzn = pdr.get_data_yahoo('AMZN')
    >>> port = pd.DataFrame({'aapl': pd.DataFrame.reset_index(aapl).iloc[:,6], 'msft':pd.DataFrame.reset_index(msft).iloc[:,6],
        'tsla': pd.DataFrame.reset_index(tsla).iloc[:,6], 'uber':pd.DataFrame.reset_index(uber).iloc[:,6],
        'amzn': pd.DataFrame.reset_index(amzn).iloc[:,6]})
    >>> port_ret = port.pct_change().dropna()
    >>> R = port_ret
    >>> w = [0.2,0.3,-0.1,0.6,0.8]
    >>> var_portfolio(w,R)

"""
    import numpy as np
    import pandas as pd

    import math as math
    import random




    weights = np.matrix(weights)
    return(float(weights.dot(R.cov()).dot(np.transpose(weights))))


# In[27]:


def HHI(weights, groups = None):

    """
    The Herfindahl Hirschman Index measures the concentration of weights using this function.
    
    Concentration of weights
    
    Parameters
    ----------
    weights : array-like,
        collection of weights of the portfolio.
    groups : dict,
        dictionary of groups.  
    
    Returns
    -------
    returns the weight concentration given the weights
    
    See Also
    --------
    port_mean
    var_portfolio
    VaR
    cVaR
    
    
    Examples
    --------
    >>> # calculate HHI
    >>> w = [0.2,0.3,-0.1,0.6,0.8]
    >>> HHI(w)


"""
    import numpy as np
    import pandas as pd

    import math as math
    import random




    if groups == None:
        return(sum(np.array(weights)**2))
    if groups != None:
        ngroups = len(groups)
        group_hhi = []
        weights = np.array(weights)
        for i in range(0,ngroups):
            group_hhi.append(sum(weights[list(groups.values())[i]]**2))
        return(group_hhi)


# In[28]:


def port_mean(weights, mu):

    """
    
    Add a constraint in portfolio_spec object
    Main function to add or update constraint in portfolio_spec object
    
    Parameters
    ----------
    weights : array-like,
        weights of the portfolio.
    mu : array-like,
        mean return of assets
    
    Returns
    -------
    Adds port mean to object portfolio_spec of the specified input. 
    
    Notes
    -----
    
    \mu_{p} = w . \mu_{r}


    
    See Also
    --------
    var_portfolio
    VaR
    cVaR
    
    
    Examples
    --------
    >>> #calculate the portfolio mean return
    >>> w = [0.2,0.3,-0.1,0.6,0.8]
    >>> port_mean(w,R)


"""
    import numpy as np
    import pandas as pd
    import math as math
    import random






    return(float(np.matrix(weights).dot(np.transpose(mu))))


# In[29]:


def VaR(R, p = 0.05):
    """
    Calculate the value at risk (VaR)
    
    This function calculates the value at risk (VaR) assuming gaussian distribution
    
    Parameters
    ----------
    R : pd.DataFrame object,
        returns dataframe
    p : float, optional
        quantile estimate
    
    Returns
    -------
    Calculates the VaR given returns
    
    See Also
    --------
    cVaR
    VaR_portfolio
        
    Examples
    --------
    >>> VaR(R,p = 0.01)
"""
    import numpy as np
    import pandas as pd
    import math as math
    import random





    r_mean = np.mean(R)
    r_std = np.std(R)
    import scipy.stats
    quantile = scipy.stats.norm.ppf(p)
    VaR = r_std*quantile
    return(VaR)


# In[30]:


def VaR_portfolio(w,R,p = 0.05, mean = True):
    
    """
    calculate the value at risk (VaR) of a portfolio
    
    This function calculates the value at risk (VaR) assuming gaussian distribution
    
    Parameters
    ----------
    w : array-like,
        list of weights of portfolio
    R : pd.DataFrame object,
        returns dataframe
    p : float, optional
        quantile estimate
    mean : bool, default = True
        mean VaR of portfolio
        
    Returns
    -------
    Calculates the VaR given portfolio weights
    
    See Also
    --------
    cVaR
    VaR_portfolio
    VaR
        
    Examples
    --------
    >>> w = [0.2,0.3,0.5]
    >>> VaR_portfolio(w,R,p = 0.01)
"""
    import numpy as np
    import pandas as pd
    import math as math
    import random




    r_mean = np.mean(R)
    r_cov = np.cov(np.transpose(np.matrix(R)))
    r_std_port = np.sqrt(np.transpose(w).dot(r_cov.dot(w)))
    import scipy.stats
    quantile = scipy.stats.norm.ppf(p)
    var_port = r_std_port*quantile
    if mean == True:
        return(np.mean(var_port))
    elif mean == False:
        return(var_port)


# In[31]:


def cVaR_portfolio(w,R,p = 0.05, mean = True):
    
    """
    Calculate the conditional value at risk (cVaR) of a portfolio
    
    This function calculates the conditional value at risk (cVaR) assuming gaussian distribution
    
    Parameters
    ----------
    w : array-like,
        list of weights of portfolio
    R : pd.DataFrame object,
        returns dataframe
    p : float, optional
        quantile estimate
    mean : bool, default = True
        mean VaR of portfolio
        
    Returns
    -------
    Calculates the VaR given portfolio weights
    
    See Also
    --------
    cVaR
    VaR_portfolio
    VaR
        
    Examples
    --------
    >>> w = [0.2,0.3,0.5]
    >>> cVaR_portfolio(w,R,p = 0.01)
"""

    import numpy as np
    import pandas as pd
    import math as math
    import random





    VaR_port = VaR_portfolio(w,R,p, mean = False)
    cvar_vec = []
    for i in range(0,R.shape[1]):
        cvar_vec.append(np.mean(R.iloc[:,i][R.iloc[:,i]<VaR_port[i]]))
    if mean == True:
        return(np.mean(cvar_vec))
    elif mean == False:
        return(cvar_vec)


# In[32]:


def group_fail(weights, groups = None, cLO = None, cUP = None, group_pos = None):
    """
    The role loops through each group and checks if the cLO or cUP for the given group has been breached.
    This is a rp_transform helper feature.
    Test if group limits have been breached.
        
    Parameters
    ----------
    weights : array-like,
        test the list of weights
    groups : dict,
        A dictionary defining the asset classes. similar in group_constraint.
    cLO : float, or array-like,
        Specifying minimum weight group constraints by numeric or vector.
    cUP : float, or array-like,
        Specifying maximum weight group constraints by numeric or vector.
    group_pos : array-like, optional
        A list that defines the number of non-zero weights for each category.
    
    Returns
    -------
    Bool returning "True" if group weights are breached
    
    
"""

    import numpy as np
    import pandas as pd
    import math as math
    import random




    if (np.any(weights) == None) or type(cLO) == None or type(cUP) == None:
        raise ValueError('One or more arguments is incorrect')
    group_count = []
    try:
        for i in range(0,len(groups.keys())):
            group_count.append(len(list(groups.values())[i]))
            group_pos = group_count
            tolerance = 1.490116e-08
        n_groups = len(groups.keys())
        group_fail = []
        for i in range(0,n_groups):
            tmp_w = np.array(weights)[list(groups.values())[i]]
            group_fail.append(((sum(tmp_w) < cLO[i]) or (sum(tmp_w) > cUP[i]) or (sum(abs(tmp_w) > tolerance) > group_pos[i])))
        return(group_fail)
    except:
        return(False)


# In[33]:


def normalize_weights(weights):
    """
    Add a constraint in portfolio_spec object
    
    Main function to add or update constraint in portfolio_spec object
    
    Parameters
    ----------
    weights : array-like,
        list of weights to normalize based on constraints.
    
    Returns
    -------
    Returns array of normalized weights. Called be optimize_portfolio when optimize_method is:
    'pso', 'dual_annealing', 'shgo', 'basinhopping', 'brute'.
"""

    import numpy as np
    import pandas as pd

    import math as math
    import random



    weights = np.array(weights)
    if not (constraints.get('weight_sum') is None):
        if not (constraints.get('weight_sum') is None):
            max_sum = constraints['weight_sum']['max_sum']
            if sum(weights) > max_sum:
                weights = (max_sum/sum(weights)) * weights
        if not (constraints.get('weight_sum') is None):
            min_sum = constraints['weight_sum']['min_sum']
            if sum(weights) < min_sum:
                weights = (min_sum/sum(weights)) * weights
    return(weights)    


# In[34]:


def generate_sequence(minimum = 0.01, maximum = 1.01, by = 0.01/1, rounding = 3):
    """
    The sequence of min<->max weights for random.
    
    Creating a series of potential weights for portfolios of random or brute force.
    
    Parameters
    ----------
    minimum : float,
        sequence minimum value   
    maximum : float,
        sequence maximum value
    by : float, optional
        number to increase the series by
    rounding : int, optional
        integer the number of decimals we can round to

    Returns
    -------
    returns a series of ranfom weights that satisfy the min-max weights constraint. 
    Default weight_seq for rp_transform
    
"""

    import numpy as np
    import pandas as pd

    import math as math
    import random



    ret = np.arange(start = round(minimum, rounding), stop = round(maximum, rounding), step = by)
    return(ret)


# In[35]:


def pos_limit_fail(weights, max_pos = None, max_pos_long= None, max_pos_short = None):
    """
    
    This is used as a rp_transform helper function to search for position limit constraints being violated.       
    
    Parameters
    ----------
    
    weights : array-like,
        test list of weights.
    max_pos : int, optional
        maximum number of non-zero-weighted assets
    max_pos_long : int, optional
        maximum number of long-position assets
    max_pos_short : int, optional
        maximum number of short-position assets
   
    
    Returns
    -------
    Returns "True" if the weights fail the pos_limit constraint. called by rp_transform. see rp_transform.
    
"""
    import numpy as np
    import pandas as pd
    import math as math
    import random




    tolerance = 1.490116e-08
    if not (max_pos == None):
        if (abs(sum(np.array(weights)) > tolerance)) > max_pos:
            return(True)
    if not (max_pos_long == None):
        if (sum(np.array(weights) > tolerance)) > max_pos_long:
            return(True)
    if not (max_pos_short == None):
        if (sum(np.array(weights) < -tolerance)) > max_pos_short:
            return(True)
    return(False)


# In[36]:


def leverage_fail(weights, leverage = None):
    """
    
    It is used by rp_transform as a helper function to test if leverage constraint is breached.
    
    Parameters
    ----------
    weights : array-like,
        test the list of weights.
    leverage : float,
        specify the leverage the portfolio must satisfy
    
    Returns
    -------
    Returns "True" if the weights fail the leverage_exposure constraint. Called by rp_transform. see rp_transform.
    
    See Also
    --------
    pos_limit_fail
      import numpy as np
    import pandas as pd


  rp_transform
    
"""


    import numpy as np
    import pandas as pd

    import math as math
    import random



    if leverage == None:
        return(False)
    elif sum(abs(np.array(weights))) > leverage:
        return(True)
    else:
        return(False)


# In[37]:


def max_sum_fail(weights, max_sum = None):
    """
    
    it is used by rp_transform as a helper function to test if max_sum constraint is breached.
    
    Parameters
    ----------
    weights : array-like,
        test the list of weights
    max_sum : float, optional
        max_sum of the weights of portfolio
   
    
    Returns
    -------
    Returns "True" if the weights fail the max_sum constraint. Called by rp_transform. see rp_transform.
    
    See Also
    --------
    fn_map, rp_transform
   
"""

    import numpy as np
    import pandas as pd

    import math as math
    import random



    if max_sum == None:
        return(False)
    elif sum(weights) > max_sum:
        return(True)
    else:
        return(False)       


# In[38]:


def min_sum_fail(weights, min_sum = None):
    
    """
    
    it is used by rp_transform as a helper function to test if min_sum constraint is breached.
    
    Parameters
    ----------
    weights : array-like,
        test the list of weights
    min_sum : float, optional
        max_sum of the weights of portfolio
   
    
    Returns
    -------
    Returns "True" if the weights fail the min_sum constraint. Called by rp_transform. see rp_transform.
    
    See Also
    --------
    fn_map, rp_transform
   
"""

    import numpy as np
    import pandas as pd

    import math as math
    import random




    if min_sum == None:
        return(False)
    elif (sum(weights) < min_sum):
        return(True)
    else:
        return False       


# In[39]:


def rp_decrease(weights, max_sum, min_box, weight_seq):
    """
    
    It is used by rp_transform as a helper function to reduce weights if 
    max_sum or min_box constraint is breached.
    
    Parameters
    ----------
    weights : array-like,
        test of list of weights
    max_sum : float,
        maximum sum of weights
    min_box : array-like,
        minimum of individual weights in a portfolio
    weight_seq : seq,
        sequence of random weights to choose from
   
    
    Returns
    -------
    returns an array of weights that satisfy the max_sum and min_box constraint. called by rp_transform.
    
    
    See Also
    --------
    rp_increase
    rp_transform
    
"""
    import numpy as np
    import pandas as pd

    import math as math
    import random




    if sum(weights) <= max_sum:
        return(weights)
    tmp_w = weights
    n_weights = len(weights)
    random_index = random.sample(range(0,n_weights), n_weights)
    i = 0
    while (sum(tmp_w) > max_sum and i < n_weights):
        cur_index = random_index[i]
        cur_val = tmp_w[cur_index]
        tmp_seq = weight_seq[(weight_seq < cur_val) & ((weight_seq) >= 
        min_box[cur_index])]
        n_tmp_seq = len(tmp_seq)
        if n_tmp_seq > 1:
            tmp_w[cur_index] = float(tmp_seq[random.randrange(0,n_tmp_seq)])
        elif n_tmp_seq == 1:
            tmp_w[cur_index] = float(tmp_seq)
        i = i + 1
    return(tmp_w)


# In[40]:


def rp_decrease_leverage(weights, max_box, min_box, leverage, weight_seq):
    """
    
    It is used by rp_transform as a helper function to redcrese leverage if 
    leverage_exposure constraint is breached.
    
    Parameters
    ----------
    weights : array-like,
        test of list of weights
    max_box : array-like,
        maximum of individual weights in a portfolio
    min_box : like,
        minimum of individual weights in a portfolio
    leverage: float,
        leverage as specified in leverage_exposure constraints
    weight_seq : seq,
        sequence of random weights to choose from
   
    
    Returns
    -------
    returns an array of weights that satisfy the leverage constraint. called by rp_transform.
    
    
    See Also
    --------
    rp_increase
    rp_transform
    
"""

    import numpy as np
    import pandas as pd

    import math as math
    import random




    tmp_w = weights
    n_weights = len(weights)
    random_index = random.sample(range(0,n_weights), n_weights)
    i = 0
    while (sum(abs(np.array(tmp_w))) > leverage) and (i < len(tmp_w)):
        cur_index = random_index[i]
        cur_val = tmp_w[cur_index]
        tmp_seq = None
        if cur_val < 0:
            tmp_seq = weight_seq[(weight_seq > cur_val) & (weight_seq <= max_box[cur_index])]
        elif cur_val > 0:
            tmp_seq = weight_seq[(weight_seq < cur_val) & (weight_seq >= 
        min_box[cur_index])]
        if not(np.any(tmp_seq) == None):
            n_tmp_seq = len(tmp_seq)
            if (n_tmp_seq > 1):
                tmp_w[cur_index] = float(tmp_seq[random.randrange(0,n_tmp_seq)])
            elif n_tmp_seq == 1:
                tmp_w[cur_index] = float(tmp_seq)
        i = i + 1
    return(tmp_w)


# In[41]:


def rp_increase(weights, min_sum, max_box, weight_seq):
    
    """
    
    It is used by rp_transform as a helper function to increase weights if 
    min_sum or max_box constraint is breached.
    
    Parameters
    ----------
    weights : array-like,
        test of list of weights
    max_sum : float,
        maximum sum of weights
    min_box : float,
        minimum of individual weights in a portfolio
    weight_seq : seq,
        sequence of random weights to choose from
   
    
    Returns
    -------
    returns an array of weights that satisfy the max_sum and min_box constraint. called by rp_transform.
    
    
    See Also
    --------
    rp_increase
    rp_transform
    
"""

    import numpy as np
    import pandas as pd
    import math as math
    import random





    if sum(weights) >= min_sum:
        return(weights)
    tmp_w = weights
    n_weights = len(weights)
    random_index = random.sample(range(0,n_weights), n_weights)
    i = 0
    while (sum(tmp_w) < max_sum and i < n_weights):
        cur_index = random_index[i]
        cur_val = tmp_w[cur_index]
        tmp_seq = weight_seq[(any(weight_seq) > cur_val) & ((weight_seq) <= 
        max_box[cur_index])]
        n_tmp_seq = len(tmp_seq)
        if n_tmp_seq > 1:
            tmp_w[cur_index] = float(tmp_seq[random.randrange(0,n_tmp_seq)])
        elif n_tmp_seq == 1:
            tmp_w[cur_index] = float(tmp_seq)
        i = i + 1
    return(tmp_w)


# In[42]:


def rp_position_limit(weights,min_box, max_box, weight_seq, max_pos = None, max_pos_long = None, max_pos_short = None):
    """
    
    It is used by rp_transform as a helper function to increase weights if 
    position_limit constraint is breached.
    
    Parameters
    ----------
    weights : array-like,
        test of list of weights
    min_box : array-like,
        maximum of individual weights in a portfolio
    min_box : array-like,
        minimum of individual weights in a portfolio
    max_pos : int, optional
        maximum position to hold in a portfolio
    max_pos_long : int, optional
        maximum long position to hold in a portfolio
    max_pos_short : int, optional
        maximum short position to hold in a portfolio
    weight_seq : seq,
        sequence of random weights to choose from
   
    
    Returns
    -------
    returns an array of weights that satisfy the position limit constraint. called by rp_transform.
    
    
    See Also
    --------
    rp_increase
    rp_transform
    rp_decrease_leverage
    
"""

    import numpy as np
    import pandas as pd
    import math as math
    import random





    tmp_w = weights
    n_weights = len(weights)
    random_index = random.sample(range(0,n_weights), n_weights)
    tolerance = 1.490116e-08
    i = 0
    while (pos_limit_fail(tmp_w, max_pos, max_pos_long, max_pos_short) and i < len(tmp_w)):
        cur_index = random_index[i]
        cur_val = tmp_w[cur_index]
        if not (max_pos_long == None):
            if (sum(np.array(tmp_w) > tolerance) > max_pos_long):
                if cur_val > tolerance:
                    tmp_seq = weight_seq[(weight_seq <= 0) & (weight_seq >= min_box[cur_index])]
                    n_tmp_seq = len(tmp_seq)
                    if n_tmp_seq > 1:
                        tmp_w[cur_index] = float(tmp_seq[random.randrange(0,n_tmp_seq)])
                    elif n_tmp_seq == 1:
                        tmp_w[cur_index] = float(tmp_seq)
        if not (max_pos_short == None):
            if (sum(np.array(tmp_w) < tolerance) > max_pos_short):
                if cur_val < tolerance:
                    tmp_seq = weight_seq[(weight_seq >= 0) & (weight_seq <= min_box[cur_index])]
                    n_tmp_seq = len(tmp_seq)
                    if n_tmp_seq > 1:
                        tmp_w[cur_index] = float(tmp_seq[random.randrange(0,n_tmp_seq)])
                    elif n_tmp_seq == 1:
                        tmp_w[cur_index] = float(tmp_seq)
        i = i + 1
    return(tmp_w)


# In[43]:


def rp_transform(w, min_sum, max_sum, min_box, max_box, groups = None, 
  cLO = None, cUP = None, group_pos = None, max_pos = None,  
  max_pos_long = None, max_pos_short = None, leverage = None, 
  weight_seq = None, max_permutations = 2000):
    """
    This function is mainly used to transform weights that dont satisfy the constraints such as:
    "box"
    "group"
    "max_pos"
    "leverage_exposure"
    See add_constraint for more deatils.
    
    Transform a list of weights to fulfill constraints
    
    Parameters
    ----------
    weights : array-like,
        list of weights to be transformed
    min_sum : float,
        minimum total of all weights of assets, default of 0.99
    max_sum : float,
        maximum total of all weights of assets, default of 1.01
    min_box : array-like,
        numeric or called list defining the minimum constraints of the weight box
    max_box : array-like
        numeric or called list defining the maximum constraints of the weight box
    groups : dict, 
        a dictionary defining the asset groups. similar to group_constraint. see group_constraint
    cLO : flaot, or array-like,
        float or list defining the minimum constraints of the weight group
    cUP : float, or array-like
        float or list defining the minimum constraints of the weight group
    group_pos : array-like, optional
        list that specifies the maximum number of non-zero-weight assets per group
    max_pos : int,
        maximum non-zero-weight assets
    max_pos_long : int
        maximum number of long (i. e. buy) position assets
    max_pos_short : int,
        maximum number of short (i. e. sell) position assets
    leverage : float,
        maximum exposure to leverage in which leverage is defined as sum(abs(weights))
    weight_seq : seq, optional
        list of seed sequence of weights. uses generate_sequence()
    max_permutations : int, optional
        integer- maximum number of iterations to try for a portfolio which is valid, default 2000
    
    Returns
    -------
    weights array that satisfy the constraints on portfolio
    
    See Also
    --------
    rp_increase
    rp_decrease
    fn_map
   
   
"""
    import numpy as np
    import pandas as pd
    import math as math
    import random





    import itertools
    
    tmp_w = w
    min_sum += -0.01
    max_sum += 0.01
    if max_pos == None:
        max_pos = len(tmp_w)
    if np.any(weight_seq) == None:
        weight_seq = generate_sequence(minimum = min(min_box), maximum = max(max_box), by = 0.02) #GenerateSequence Function
    if not (max_pos == None) or not(group_pos == None) or not(max_pos_long == None) or not(max_pos_short == None) and np.any(weight_seq == 0):
        tolerence = 1.490116e-08
    permutation = 0
    while ((min_sum_fail(tmp_w, min_sum) or max_sum_fail(tmp_w, 
    max_sum) or leverage_fail(tmp_w, leverage) or pos_limit_fail(tmp_w, 
    max_pos, max_pos_long, max_pos_short) or (np.any(group_fail(tmp_w, 
    groups, cLO, cUP)) == True)) and (permutation < max_permutations)):
        permutation = permutation + 1
        n_weights = len(tmp_w)
        random_index = random.sample(range(0,n_weights), max_pos)
        full_index = range(0,n_weights)
        not_index = list(set(full_index).difference(set(list(random_index))))
        for k in range(0,len(not_index)):
            tmp_w[not_index[k]] = 0
        if (min_sum_fail(tmp_w, min_sum)):
            tmp_w = rp_increase(weights = tmp_w, min_sum = min_sum, 
        max_box = max_box, weight_seq = weight_seq)
        if (max_sum_fail(tmp_w, max_sum)):
            tmp_w = rp_decrease(weights = tmp_w, max_sum = max_sum, min_box = min_box, weight_seq = weight_seq)
        if (leverage_fail(tmp_w, leverage)):
            tmp_w = rp_decrease_leverage(weights = tmp_w, max_box = max_box, min_box = min_box, leverage = leverage, weight_seq = weight_seq)
        if (pos_limit_fail(tmp_w, max_pos, max_pos_long, max_pos_short)):
            tmp_w = rp_position_limit(weights = tmp_w, min_box = min_box, max_box = max_box, max_pos = max_pos, max_pos_long = max_pos_long, max_pos_short = max_pos_short, weight_seq = weight_seq)
        if groups != None:
            if (any(group_fail(tmp_w, groups, cLO, cUP, group_pos))):
                n_groups = len(groups)
                tmp_group_i = []
                for j in range(0,n_groups):
                    j_idx = list(groups.values())[j]
                    tmp_group_w = []
                    tmp_min_box = []
                    tmp_max_box = []
                    for js in j_idx:
                        tmp_group_w.append(tmp_w[js])
                        tmp_min_box.append(min_box[js])
                        tmp_max_box.append(max_box[js])
                    tmp_group_i.append([rp_transform(w = tmp_group_w, 
                                        min_sum = cLO[j], max_sum = cUP[j], min_box = tmp_min_box, 
                                            max_box = tmp_max_box, max_permutations = 2000)])
                tmp_group_i = list(itertools.chain(*tmp_group_i))
                tmp_group_i = list(itertools.chain(*tmp_group_i))
                for k in range(0,n_groups):
                    j_idx = list(groups.values())[k]
                    tmp_w[k] = tmp_group_i[k]
                                            #group constraint

    portfolio = tmp_w
    if (sum(portfolio) < min_sum or sum(portfolio) > max_sum):
        raise ValueError('Impossible portfolio created. Try relaxing constraints')
    return(portfolio)


# In[44]:


def fn_map(weights, portfolio, relax = False, verbose = False):
    """
    
    This function transforms list of weights that does not meet the portfolio constraints to an array that meets
    the constraints. relax argument (default = False) if True gives the function permission to transform weights 
    if needed
    
    Parameters
    ----------
    weights : array-like,
        list of initial weights.
    portfolio : portfolio_spec,
        an object of class portfolio_spec.
    relax : bool, default = False
        bool to enable or disable constraints.
    verbose : bool, default = False
        bool to enable or disable messages.
   
    
    Returns
    -------
    returns the array of weights tranformed by the function and that does satisfy the constraints.
    called by optimize_portfolio if optimize_method = 'DEoptim' or 'pso'
   
"""

    import numpy as np
    import pandas as pd
    import math as math
    import random




    nassets = len(portfolio.assets)
    constraints = get_constraints(portfolio)
    if not (constraints.get('weight_sum') == None):
        min_sum = constraints['weight_sum']['min_sum']
        max_sum = constraints['weight_sum']['max_sum']     
    if not (constraints.get('leverage') == None):
        min_sum = constraints['leverage']['min_sum']
        max_sum = constraints['leverage']['max_sum']     
    if not (constraints.get('weight') == None):
        min_sum = constraints['weight']['min_sum']
        max_sum = constraints['weight']['max_sum']     
    if not (constraints.get('full_investment') == None):
        min_sum = constraints['full_investment']['min_sum']
        max_sum = constraints['full_investment']['max_sum']     
    if not (constraints.get('dollar_neutral') == None):
        min_sum = constraints['dollar_neutral']['min_sum']
        max_sum = constraints['dollar_neutral']['max_sum']


    if not(min_sum == None and max_sum == None):
        if (max_sum - min_sum) < 0.02:
            min_sum = min_sum - 0.01
            max_sum = max_sum + 0.01
   
    try:
        if not (constraints.get('box') == None):
            minimum = constraints['box']['minimum']
            maximum = constraints['box']['maximum']
        if not (constraints.get('long_only') == None):
            minimum = constraints['long_only']['minimum']
            maximum = constraints['long_only']['maximum']
    except:
        minimum = np.repeat(0, len(portfolio.assets))
        maximum = np.repeat(1, len(portfolio.assets))
    
    if np.array(portfolio.weight_seq) == None:
        weight_seq = generate_sequence(minimum = min(minimum), maximum = max(maximum), by = 0.02) #generateSequence function
        weight_seq = np.array(weight_seq)
    
    
    weight_seq = np.array(weight_seq)
    minimum = np.array(minimum)
    maximum = np.array(maximum)


    if not(constraints.get('group') == None):
        groups = constraints['group']['groups']
        cLO = constraints['group']['group_min']
        cUP = constraints['group']['group_max'] #add group_pos assuming None for for
        tmp_cLO = cLO
        tmp_cUP = cUP
        group_pos = None
    else:
        groups = None
        cLO = None
        cUP = None #add group_pos assuming None for for
        tmp_cLO = None
        tmp_cUP = None
        group_pos = None



    tmp_min = np.array(minimum)
    tmp_max = np.array(maximum)
    tmp_weights = np.array(weights)
    tolerance = 1.490116e-08


    if not(constraints.get('turnover') == None):
        turnover_target = constraints['turnover']['turnover_target']
    else:
        turnover_target = None
    if not(constraints.get('diversification') == None):
        div_target = constraints['diversification']['diversification_target']
    else:
        div_target = None
    if not (constraints.get('position_limit') == None):
        max_pos = constraints['position_limit']['max_pos']
        max_pos_long = constraints['position_limit']['max_pos_long']
        max_pos_short = constraints['position_limit']['max_pos_short']
        tmp_max_pos = max_pos
        tmp_max_pos_long = max_pos_long
        tmp_max_pos_short = max_pos_short
    else:
        max_pos = None
        max_pos_long = None
        max_pos_short = None
        tmp_max_pos = None
        tmp_max_pos_long = None
        tmp_max_pos_short = None
    if not (constraints.get('leverage') == None):
        leverage = constraints['leverage']['leverage']
        tmp_leverage = leverage
    else:
        leverage = None
        tmp_leverage = None

    if not(min_sum == None and max_sum == None):
            if not (sum(np.array(tmp_weights)) >= min_sum and sum(np.array(tmp_weights)) <= np.array(max_sum)):
                try:
                    tmp_weights = rp_transform(w = tmp_weights, 
                                        min_sum = min_sum, max_sum = max_sum, min_box = tmp_min, 
                                        max_box = tmp_max, groups = None, cLO = None, 
                                        cUP = None, max_pos = None, group_pos = None, 
                                        max_pos_long = None, max_pos_short = None, leverage = tmp_leverage, 
                                        weight_seq = weight_seq, max_permutations = 1000)
                except:
                    tmp_weights = weights
            if not (np.all(tmp_weights>= np.array(tmp_min)) and np.all(tmp_weights <= np.array(tmp_max))):
                try:
                    tmp_weights = rp_transform(w = tmp_weights, 
                                    min_sum = min_sum, max_sum = max_sum, min_box = tmp_min, 
                                    max_box = tmp_max, groups = None, cLO = None, 
                                    cUP = None, max_pos = None, group_pos = None, 
                                    max_pos_long = None, max_pos_short = None, leverage = tmp_leverage, 
                                    weight_seq = weight_seq, max_permutations = 1000)
                except:
                    tmp_weights = weights
            if relax == True:
                i = 0
                while ((sum(tmp_weights) < min_sum or sum(tmp_weights) > max_sum or np.any(tmp_weights < tmp_min) or np.any(tmp_weights > tmp_max))) and i < 5:
                    if np.any(tmp_weights < tmp_min):
                        for j in range(0,len(tmp_min)):
                            if tmp_weights[j]<tmp_min[j]:
                                tmp_min[j] = tmp_min[j] - float(np.random.uniform(0.01,0.05,1))
                    if np.any(tmp_weights > tmp_max):
                        for k in range(0,len(tmp_max)):
                            if tmp_weights[k]>tmp_max[k]:
                                tmp_max[k] = tmp_max[k] + float(np.random.uniform(0.01,0.05,1))
                    try:
                        tmp_weights = rp_transform(w = tmp_weights, 
                        min_sum = min_sum, max_sum = max_sum, 
                        min_box = tmp_min, max_box = tmp_max, 
                        groups = None, cLO = None, cUP = None, 
                        max_pos = None, group_pos = None, max_pos_long = None, 
                        max_pos_short = None, leverage = tmp_leverage, 
                        weight_seq = weight_seq, max_permutations = 3000)
                    except:
                        tmp_weights = weights
                    i += 1
                if np.all(tmp_weights == weights):
                    tmp_min = minimum
                    tmp_max = maximum

                        
    #group_transform
    
    
    if not (np.all(groups) == None and np.all(cLO) == None and np.all(cUP) == None):
        if np.any(group_fail(tmp_weights, groups, tmp_cLO, tmp_cUP)):
            try:
                tmp_weights = rp_transform(w = tmp_weights, 
            min_sum = min_sum, max_sum = max_sum, min_box = tmp_min, 
            max_box = tmp_max, groups = groups, cLO = tmp_cLO, 
            cUP = tmp_cUP, max_pos = None, group_pos = group_pos, 
            max_pos_long = None, max_pos_short = None, leverage = tmp_leverage, 
            weight_seq = weight_seq, max_permutations = 1000)
            except:
                tmp_weights = weights
        if relax == True:
            i = 0
            while (((sum(tmp_weights) < min_sum or sum(tmp_weights) > max_sum) or (np.any(tmp_weights < tmp_min) or np.any(tmp_weights > tmp_max)) or np.any(group_fail(tmp_weights, groups, tmp_cLO, tmp_cUP, group_pos))) and i < 5):
                if np.any(group_fail(tmp_weights, groups, tmp_cLO, tmp_cUP, group_pos)):
                    for j in range(0,len(tmp_cLO)):
                        tmp_cLO[j] = tmp_cLO[j] - float(np.random.uniform(0.01,.05,1))
                if np.any(group_fail(tmp_weights, groups, tmp_cLO, tmp_cUP, group_pos)):
                    for k in range(0,len(tmp_cUP)):
                        tmp_cUP[k] = tmp_cUP[k] + float(np.random.uniform(0.01,.05,1))
                try:
                    tmp_weights = rp_transform(w = tmp_weights, 
                  min_sum = min_sum, max_sum = max_sum, 
                  min_box = tmp_min, max_box = tmp_max, 
                  groups = groups, cLO = tmp_cLO, cUP = tmp_cUP, 
                  max_pos = None, group_pos = group_pos, 
                  max_pos_long = None, max_pos_short = None, 
                  leverage = tmp_leverage, weight_seq = weight_seq, 
                  max_permutations = 3000)
                except:
                    tmp_weights = weights
                i += 1
            if np.all(tmp_weights == weights):
                tmp_cLO = cLO
                tmp_cUP = cUP





                #Max_pos Transform
                

    
    if not (max_pos == None or max_pos_long == None or max_pos_short == None):
        if (pos_limit_fail(tmp_weights, tmp_max_pos, tmp_max_pos_long, tmp_max_pos_short)):
            try:
                tmp_weights = rp_transform(w = tmp_weights, 
                min_sum = min_sum, max_sum = max_sum, min_box = tmp_min, 
                max_box = tmp_max, groups = groups, cLO = tmp_cLO, 
                cUP = tmp_cUP, max_pos = tmp_max_pos, group_pos = group_pos, 
                max_pos_long = tmp_max_pos_long, max_pos_short = tmp_max_pos_short, 
                leverage = tmp_leverage, weight_seq = weight_seq, 
                max_permutations = 1000)
            except:
                tmp_weights = weights
        if relax == True:
            i = 0
            while (pos_limit_fail(tmp_weights, tmp_max_pos, tmp_max_pos_long, tmp_max_pos_short) and (i < 5)):
                if not (tmp_max_pos == None):
                    tmp_max_pos = min(nassets, tmp_max_pos+1)
                if not (tmp_max_pos_long == None):
                    tmp_max_pos_long = min(nassets, tmp_max_pos_long+1)
                if not (tmp_max_pos_short == None):
                    tmp_max_pos_short = min(nassets, tmp_max_pos_short+1)
                try:
                    tmp_weights = rp_transform(w = tmp_weights, 
                  min_sum = min_sum, max_sum = max_sum, 
                  min_box = tmp_min, max_box = tmp_max, 
                  groups = groups, cLO = tmp_cLO, cUP = tmp_cUP, 
                  max_pos = tmp_max_pos, group_pos = group_pos, 
                  max_pos_long = tmp_max_pos_long, max_pos_short = tmp_max_pos_short, 
                  leverage = tmp_leverage, weight_seq = weight_seq, 
                  max_permutations = 3000)
                except:
                    tmp_weights = weights
                i += 1


    #Leverage Transform

    if not (tmp_leverage == None):
        if (sum(abs(np.array(tmp_weights))) > tmp_leverage):
            try:
                tmp_weights = rp_transform(w = tmp_weights, 
            min_sum = min_sum, max_sum = max_sum, min_box = tmp_min, 
            max_box = tmp_max, groups = groups, cLO = tmp_cLO, 
            cUP = tmp_cUP, max_pos = tmp_max_pos, group_pos = group_pos, 
            max_pos_long = tmp_max_pos_long, max_pos_short = tmp_max_pos_short, 
            leverage = tmp_leverage, weight_seq = weight_seq, 
            max_permutations = 1000)
            except:
                tmp_weights = weights
        if relax == True:
            i = 0
            while (sum(abs(np.array(tmp_weights))) > tmp_leverage and (i <= 5)):
                tmp_leverage = tmp_leverage *1.01
                try:
                    tmp_weights = rp_transform(w = tmp_weights, 
                  min_sum = min_sum, max_sum = max_sum, 
                  min_box = tmp_min, max_box = tmp_max, 
                  groups = groups, cLO = tmp_cLO, cUP = tmp_cUP, 
                  max_pos = tmp_max_pos, group_pos = group_pos, 
                  max_pos_long = tmp_max_pos_long, max_pos_short = tmp_max_pos_short, 
                  leverage = tmp_leverage, weight_seq = weight_seq, 
                  max_permutations = 3000)
                except:
                    tmp_weights = weights
                i += 1   
                
                
                
    return(dict(zip(['weights', 'minimum', 'maximum', 'cLO', 'cUP', 'max_pos', 'max_pos_long', 'max_pos_short', 'leverage', 'min_sum', 'max_sum'],
                   [tmp_weights, tmp_min, tmp_max, tmp_cLO, tmp_cUP, tmp_max_pos, tmp_max_pos_long, tmp_max_pos_short, tmp_leverage, min_sum, max_sum])))


# In[45]:


def extract_weights(portfolio):
    """
    
    Function to extract optimal weights.
    
    Parameters
    ----------
    portfolio : portfolio_spec,
        an object of class portfolio_spec.  
    
    Returns
    -------
    array of optimal weights
    
    See Also
    --------
    extract_objective_measure
    extract_groups
    
    Examples
    --------
    >>> extract_weights(portfolio)
    NOTE: extract_weights will only work after calling optimize_portfolio on portfolio_spec
    
    """
    import numpy as np
    import pandas as pd

    import math as math
    import random





    if not np.all(portfolio.weights) == None:
        return(portfolio.weights)
    elif portfolio.weights == None:
        raise ValueError('Portfolio_spec object does not have any weights. Please run optimize_portfolio before extracting weights')


# In[46]:


def extract_objective_measure(R, portfolio, **kwargs):
    """
    
    Function to extract objective measures of optimal weights.
    
    Parameters
    ----------
    R : pd.DataFrame,
        dataframe of returns of assets in portfolio
    portfolio : portfolio_spec,
        an object of class portfolio_spec.  
    kwargs : additional arguments, optional
        any additional argument to be passed.
    
    Returns
    -------
    dictionary of objective measures as specified in the portfolio_spec object
    
    See Also
    --------
    extract_weights
    extract_objective_measure
    extract_groups
    
    Examples
    --------
    >>> extract_objective_measure(R, portfolio)
    NOTE: extract_objective_measure will only work after calling optimize_portfolio on portfolio_spec
    
    """

    import numpy as np
    import pandas as pd
    import math as math
    import random





    if not (portfolio.weights == None):
        w = portfolio.weights
    elif (portfolio.weights == None):
        raise TypeError('Please specify portfoio weights in portfolio_spec_object.')
    tmp_objective_measure = constrained_objective(w, R, portfolio, trace = True)['objective_measures']
    return(tmp_objective_measure)


# In[47]:


def extract_groups(portfolio):
    """
    
    Function to extract groups from portfolio_spec object.
    
    Parameters
    ----------
    portfolio : portfolio_spec,
        an object of class portfolio_spec.  
    
    Returns
    -------
    dictionary of groups if group_constraint is specified.
    
    See Also
    --------
    extract_weights
    extract_objective_measure
    extract_groups
    
    Examples
    --------
    >>> extract_groups(portfolio)
    NOTE: extract_groups will only work after calling optimize_portfolio on portfolio_spec
    
    """


    import numpy as np
    import pandas as pd
    import math as math
    import random




    constraints = get_constraints(portfolio)
    if not (constraints.get('group') == None):
        groups = constraints['group']['groups']
        group_vals = list(groups.values())
        weights = np.array(portfolio.weights)
        group_weights = []
        for i in range(0,len(group_vals)):
            group_weights.append(sum(weights[group_vals[i]]))
    if not (portfolio.category_labels == None):
        category= portfolio.category_labels
        category_names = list(category.keys())
        ncategory = len(category)
        category_vals = list(category.values())
        weights = np.array(portfolio.weights)
        category_weights = []
        for i in range(0, len(category_vals)):
            category_weights.append(sum(weights[category_vals[i]]))
    tmp_group_w = list([{'groups':groups}, {'group_weights':dict(zip(list(groups.keys()), group_weights))}, {'category_labels':category_labels}, {'category_weights':dict(zip(list(category_labels.keys()), category_weights))}])
    return(tmp_group_w)   


# In[48]:


def constrained_objective(w, R, portfolio, trace = False,
                          normalize = False, storage = False, verbose= False,penalty = 10000, *kwargs, **args):
    """
    Add a constraint in portfolio_spec object
    
    Main function to add or update constraint in portfolio_spec object
    
    Parameters
    ----------
    w : array-like,
        weights to test
    R : pd.DataFrame,
        dataframe of returns of assets in portfolio
    trace : bool, default = False
        bool to enable or disable constraints.
    normalize : bool, default = False
        bool to specify if weights should be normalized first. see normalize_weights.
    verbose : bool, default = False
        bool to enable or disable verbose argument.
    penalty : int, optional
        int value specifying he penalty if constraint or objective is breached.
    kwargs : additional key word arguments, optional
        any additional constraint argument to be passed.
    kwargs : additional arguments, optional
        any additional argument to be passed.
        
        
    Returns
    -------
    returns a float of total penalty given a weight. if trace = True, additional objective measure will be returned
    in the form of dictionary.
    
    See Also
    --------
    rp_transform
    optimize_portfolio
    
    Notes
    -----
    constrained_objective is the main function that is called by 
    optimize_weights to optimze the constraint and objectives provided by the portfolio.
    
    Loosely speaking, constrained_objective is very similar to an information criteria.

"""    
    import numpy as np
    import pandas as pd
    import math as math
    import random





    if len(R.columns)>len(w):
        R = R.iloc[:,0:len(w)]
    if penalty < 10000:
        penalty = 10000
    N = len(w)
    T = R.shape[0]
    optimize_method = ''
    init_weights = w
    verbosoe = False
    constraints = get_constraints(portfolio = portfolio)
    if N != len(portfolio.assets):
        raise ValueError('The len of portfolio assets is not equal to the weights list')
    out = 0
    
    if not (constraints == None):
        if not (constraints.get('weight') == None):
            min_sum = constraints['weight']['min_sum']
            max_sum = constraints['weight']['min_sum']
        elif not (constraints.get('leverage') == None):
            min_sum = constraints['leverage']['min_sum']
            max_sum = constraints['leverage']['min_sum']

        elif not (constraints.get('weight_sum') == None):
            min_sum = constraints['weight_sum']['min_sum']
            max_sum = constraints['weight_sum']['min_sum']

        elif not (constraints.get('full_investment') == None):
            min_sum = constraints['full_investment']['min_sum']
            max_sum = constraints['full_investment']['min_sum']

        elif not (constraints.get('dollar_neutral') == None):
            min_sum = constraints['dollar_neutral']['min_sum']
            max_sum = constraints['dollar_neutral']['min_sum']
        else:
            min_sum = None
            max_sum = None


        if not (constraints.get('box') == None):
            minimum = constraints['box']['minimum']
            maximum = constraints['box']['maximum']
        elif not (constraints.get('long_only') == None):
            minimum = constraints['long_only']['minimum']
            maximum = constraints['long_only']['maximum']
        else:
            minimum = None
            maximum = None


        #add normalize function (skipped for now)

        if normalize == True:
            w = fn_map(weights = w, portfolio = portfolio)['weights']
        else:

    #add normalize function (skipped for now)
            w = np.array(w)
            if not (max_sum is None):
                if sum(w) > max_sum:
                    out += penalty*(sum(w)-max_sum)
            if not (min_sum is None):
                if sum(w) < min_sum:
                    out += penalty*(min_sum-sum(w))



        if not (maximum is None):
            if np.any(np.array(w) > np.array(maximum)):
                tmp_val = []
                for i in range(0,N):
                    if w[i]>maximum[i]:
                        tmp_val.append(w[i]-maximum[i])
                out += sum(tmp_val) * penalty
        if not (minimum is None):
            if np.any(np.array(w) < np.array(minimum)):
                tmp_val = []
                for i in range(0,N):
                    if w[i]<minimum[i]:
                        tmp_val.append(minimum[i]-w[i])
                out += sum(tmp_val) * penalty


        #Group Constraint Penalty
        if not (constraints.get('group') is None):
            groups = constraints['group']['groups']
            cLO = constraints['group']['group_min']
            cUP = constraints['group']['group_max']
            if np.any(group_fail(weights = w,groups = groups, cLO = cLO, cUP = cUP)):
                n_groups = len(groups)
                for i in range(0,n_groups):
                    tmp_w = np.array(w)[list(groups.values())[i]]
                    if sum(tmp_w) < cLO[i]:
                        out += penalty * (cLO[i] - sum(tmp_w))
                    if sum(tmp_w) > cUP[i]:
                        out += penalty * (sum(tmp_w) - cUP[i])


        if not (constraints.get('position_limit') is None):
            max_pos = constraints['position_limit']['max_pos']
            tolerence = 1.490116e-08
            mult = 1
            nzassets = sum(np.abs(w)>tolerence)
            if nzassets > max_pos:
                out += penalty * mult * (nzassets - max_pos)    
            #diversification Constraint Penalty
        if not (constraints.get('diversification') is None):
            div_target = constraints['diversification']['diversification_target']
            div = diversification(np.array(w))
            mult = 1
            if div < div_target * 0.95 or div > div_target * 1.05:
                out += penalty * mult * abs(div - div_target)

        #Turnover Constraint
        if not (constraints.get('turnover') is None):
            turnover_target = constraints['turnover']['turnover_target']
            to = turnover(np.array(w))
            mult = 1
            if (to < turnover_target * 0.95) or (to > turnover_target * 1.05):
                out += penalty * mult * abs(to - turnover_target)


        #Return Constraint
        if not (constraints.get('return') is None):
            return_target = constraints['return']['return_target']
            mu = np.matrix(port_ret.mean())
            mean_return = port_mean(weights = np.matrix(w), mu = mu)
            mult = 1
            out += penalty * mult * abs(mean_return - return_target)
            out = float(out)


        #Factor Exposure Constraint
        if not (constraints.get('factor_exposure') is None):
            t_B = np.transpose(constraints['factor_exposure']['B'])
            lower = constraints['factor_exposure']['lower']
            upper = constraints['factor_exposure']['upper']
            if type(lower) == float:
                lower = [constraints['factor_exposure']['lower']]
                upper = [constraints['factor_exposure']['upper']]

            mult = 1
            for i in range(0,t_B.shape[0]):
                tmp_exp = float(np.matrix(w).dot(np.transpose(t_B[i][:])))
                if tmp_exp < lower[i]:
                    out += penalty * mult * (lower[i] - tmp_exp)
                if tmp_exp > upper[i]:
                    out += penalty * mult * (tmp_exp - upper[i])


        #Transaction Cost Constraint
        if not (constraints.get('transaction') is None):
            tc = sum(abs(np.array(np.array(w)) - np.repeat(1/len(portfolio.assets), len(portfolio.assets)))* constraints['transaction']['ptc'])
            mult = 1
            out += mult * tc

        #Leverage Constraint Penalty
        if not (constraints.get('leverage_exposure') is None):
            if sum(abs(np.array(w))) > constraints['leverage_exposure']['leverage']:
                mult = 1/100
                out += penalty * mult * (abs(sum(abs(np.array(w))) - constraints['leverage_exposure']['leverage']))




      
    
    #assuming verbose = False for now...
    
    #Objectives Penalty Functions
    if portfolio.objectives == None:
        return(out)
    
    
    import math as math

    if out == math.inf or out == None or out == math.nan:
        raise ValueError('Na or NaN produced from constraints/objective')
    elif storage == True or trace == True:
        tmp_ret = []
    for objective in portfolio.objectives:
        if portfolio.objectives[objective]['enabled'] == True:
            tmp_measure = []
            multiplier = portfolio.objectives[objective]['multiplier']
            if objective == 'return':
                if portfolio.objectives[objective]['name'] == 'mean':
                    mu = np.matrix(R.mean())
                    tmp_measure = port_mean(weights = np.matrix(w), mu = mu)
                elif portfolio.objectives[objective]['name'] == 'median':
                    mu = np.matrix(R.median())
                    tmp_measure = port_mean(weights = np.matrix(w), mu = mu)
                try:
                    if (type(portfolio.objectives[objective]['name']) == dict):
                        arguments = list(portfolio.objectives[objective]['arguments'].values())
                        tmp_measure = portfolio.objectives[objective]['name'][list(portfolio.objectives[objective]['name'].keys())[0]](*arguments)
                except:
                    tmp_measure = 0.0
                    
            if objective == 'risk' or objective == 'portfolio_risk':
                if portfolio.objectives[objective]['name'] == 'std' or portfolio.objectives[objective]['name'] == 'stdev':
                    tmp_measure = float(np.sqrt(np.matrix(w).dot(np.matrix(R.cov())).dot(np.transpose(np.matrix(w)))))
                if portfolio.objectives[objective]['name'] == 'VaR':
                    if not (portfolio.objectives['risk']['arguments'].get('p') == None):
                        p = portfolio.objectives['risk']['arguments']['p']
                    else:
                        p = 0.05
                    tmp_measure = VaR_portfolio(w,R,p) # add Value at risk function. Skipped for now
                if portfolio.objectives[objective]['name'] == 'cVaR':
                    if not (portfolio.objectives['risk']['arguments'].get('p') == None):
                        p = portfolio.objectives['risk']['arguments']['p']
                    else:
                        p = 0.05
                    tmp_measure = cVaR_portfolio(w,R,p) # add cVaR function. Skipped for now
                    
                try:
                    if (type(portfolio.objectives[objective]['name']) == dict):
                        arguments = list(portfolio.objectives[objective]['arguments'].values())
                        tmp_measure = portfolio.objectives[objective]['name'][list(portfolio.objectives[objective]['name'].keys())[0]](*arguments)
                except:
                    tmp_measure = 0.0
                   
            if objective == 'turnover':
                tmp_measure = turnover(w)
            if objective == 'minmax':
                try:
                    if (type(portfolio.objectives[objective]['name']) == dict):
                        arguments = list(portfolio.objectives[objective]['arguments'].values())
                        tmp_measure = portfolio.objectives[objective]['name'][list(portfolio.objectives[objective]['name'].keys())[0]](*arguments)
                except:
                    tmp_measure = 0.0

            if objective == 'weight_concentration':
                tmp_measure = HHI(w,portfolio.objectives['weight_concentration']['conc_groups'])
            if objective == 'risk_budget':
                if portfolio.objectives[objective]['name'] == 'risk_budget':
                    if not (portfolio.objectives['risk_budget']['arguments'].get('p') == None):
                        p = portfolio.objectives['risk_budget']['arguments']['p']
                    else:
                        p = 0.05
                        tmp_measure = cVaR_portfolio(w,R,p)
                #Custom function support
                try:
                    if (type(portfolio.objectives[objective]['name']) == dict):
                        arguments = list(portfolio.objectives[objective]['arguments'].values())
                        tmp_measure = portfolio.objectives[objective]['name'][list(portfolio.objectives[objective]['name'].keys())[0]](*arguments)
                except:
                    tmp_measure = 0.0
            #Update modified sharpe and complex metrics
            if objective == 'performance_metrics':
                if portfolio.objectives[objective]['name'] == 'sharpe':
                    sd = float(np.sqrt(np.matrix(w).dot(np.matrix(R.cov())).dot(np.transpose(np.matrix(w)))))
                    if not (portfolio.objectives['performance_metrics']['arguments'].get('rf') == None):
                        rf = portfolio.objectives['performance_metrics']['arguments']['rf']
                    else:
                        rf = 0
                    mu = np.matrix(R.mean())
                    r_premium = port_mean(weights = np.matrix(w), mu = mu)-rf
                    tmp_measure = r_premium/sd
                if portfolio.objectives[objective]['name'] == 'treynor':
                    tmp_beta = []
                    for i in range(0,R.shape[1]):
                        tmp_beta.append(np.cov(np.transpose(R.iloc[:,i]), Rb)[0,1]/np.var(Rb))
                    beta = float(np.matrix(tmp_beta).dot(np.transpose(np.matrix(w))))
                    
                    if not (portfolio.objectives['performance_metrics']['arguments'].get('rf') == None):
                        rf = portfolio.objectives['performance_metrics']['arguments']['rf']
                    else:
                        rf = 0 
                    mu = np.matrix(R.mean())
                    r_premium = port_mean(weights = np.matrix(w), mu = mu)[0]-rf
                    tmp_measure = r_premium/beta
                if portfolio.objectives[objective]['name'] == 'starr':
                    if not (portfolio.objectives['performance_metrics']['arguments'].get('rf') == None):
                        rf = portfolio.objectives['performance_metrics']['arguments']['rf']
                    else:
                        rf = 0
                    if not (portfolio.objectives['performance_metrics']['arguments'].get('p') == None):
                        p = portfolio.objectives['performance_metrics']['arguments']['p']
                    else:
                        p = 0.05
                    tmp_measure = starr_ratio(w, R, p = p, rf = rf)
                try:
                    if (type(portfolio.objectives[objective]['name']) == dict):
                        arguments = list(portfolio.objectives[objective]['arguments'].values())
                        tmp_measure = portfolio.objectives[objective]['name'][list(portfolio.objectives[objective]['name'].keys())[0]](*arguments)
                except:
                    tmp_measure = 0.0

#use getattr in future
        
        tmp_measure = np.array(tmp_measure)
        if objective == 'return':
            if not (portfolio.objectives[objective]['target'] is None):
                out += penalty*abs(portfolio.objectives[objective]['multiplier'])*abs(tmp_measure - portfolio.objectives[objective]['target'])
            out += portfolio.objectives[objective]['multiplier']*tmp_measure
        if objective == 'portfolio_risk' or objective == 'risk':
            if not (portfolio.objectives[objective]['target'] is None):
                out += penalty*abs(portfolio.objectives[objective]['multiplier'])*abs(tmp_measure - portfolio.objectives[objective]['target'])
            out += portfolio.objectives[objective]['multiplier']*tmp_measure
        if objective == 'turnover':
            if not (portfolio.objectives[objective]['target'] is None):
                out += penalty*abs(portfolio.objectives[objective]['multiplier'])*abs(tmp_measure - portfolio.objectives[objective]['target'])
            out += portfolio.objectives[objective]['multiplier']*tmp_measure
        if objective == 'minmax':
            if not (portfolio.objectives[objective]['target'] is None):
                out += penalty*abs(portfolio.objectives[objective]['multiplier'])*abs(tmp_measure - portfolio.objectives[objective]['target'])
            out += portfolio.objectives[objective]['multiplier']*tmp_measure
        if objective == 'weight_concentration':
            if (type(portfolio.objectives[objective]['conc_aversion']) == float) and (portfolio.objectives[objective]['conc_groups'] == None):
                out += penalty * portfolio.objectives[objective]['conc_aversion'] * tmp_measure
            if (type(portfolio.objectives[objective]['conc_aversion']) == list) and not (portfolio.objectives[objective]['conc_groups'] == None):
                    out += penalty * portfolio.objectives[objective]['multiplier'] * sum(tmp_measure * np.array(portfolio.objectives[objective]['conc_aversion']))
        if objective == 'performance_metrics':
            if not (portfolio.objectives[objective]['target'] is None):
                out += penalty*abs(portfolio.objectives[objective]['multiplier'])*abs(tmp_measure - portfolio.objectives[objective]['target'])
            out += portfolio.objectives[objective]['multiplier']*tmp_measure
 
                    #Risk Budget
        if objective == 'risk_budget':
            if not (portfolio.objectives[objective]['target'] is None):
                out += penalty*abs(portfolio.objectives[objective]['multiplier'])*abs(tmp_measure - portfolio.objectives[objective]['target'])
        
        #end of objective penalty
        if trace == True:
            try:
                tmp_ret.append([objective, tmp_measure])
            except:
                tmp_ret = tmp_ret

                
    if verbose == True:
        print('weights : {}'.format(w))
        print('out : {}'.format(out))
        print('objective_measures : {}'.format(tmp_ret))
    
    if storage == True or trace == True:
        return({'out':out, 'objective_measures':tmp_ret, 'weights':w})
    elif storage == False or trace == False:
        return(float(out))


# In[49]:


def chart_weights(portfolio):
    
    """
    
    Function to plot weights of optimal weights.
    
    Parameters
    ----------
    portfolio : portfolio_spec,
        an object of class portfolio_spec.  
    
    Returns
    -------
    matplotlib.lines.Line2D
    
    See Also
    --------
    chart_group_weights
    chart_efficient_frontier
    
    
    Examples
    --------
    >>> chart_weights(portfolio)
    NOTE: chart_group_weights will only work after call optimize_portfolio on portfolio_spec
    
    """
    import numpy as np
    import pandas as pd
    import math as math
    import random





    import matplotlib.pyplot as plt
    w = portfolio.weights
    constraints = get_constraints(portfolio)
    if not (constraints.get('box') == None):
        minimum = constraints['box']['minimum']
        maximum = constraints['box']['maximum']
    elif not (constraints.get('long_only') == None):
        minimum = constraints['long_only']['minimum']
        maximum = constraints['long_only']['maximum']
    else:
        minimum = None
        maximum = None
    if np.any(minimum) == None or np.any(maximum) == None or np.any(w) == None:
        raise ValueError('One of the requirement if missing')
    
    nassets = len(portfolio.assets)
    assets = portfolio.assets
    plt.plot(w, 'r>', linewidth = 2, linestyle = '--', c = 'b')
    plt.plot(minimum, 'r*', linewidth = 2, linestyle = ':', c = 'gray')
    plt.plot(maximum, 'r*', linewidth = 2, linestyle = ':', c = 'gray')
    plt.ylabel('Weights')
    plt.xticks(list(range(0,nassets)), labels = assets)


# In[50]:


def chart_group_weights(portfolio):
    """
    
    Function to plot individual group weights of optimal weights.
    
    Parameters
    ----------
    portfolio : portfolio_spec,
        an object of class portfolio_spec.  
    
    Returns
    -------
    matplotlib.lines.Line2D
    
    See Also
    --------
    chart_group_weights
    chart_efficient_frontier
    
    
    Examples
    --------
    >>> chart_group_weights(portfolio)
    NOTE: chart_group_weights will only work after call optimize_portfolio on portfolio_spec
"""


    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import math as math
    import random






    constraints = get_constraints(portfolio)
    group_min = constraints['group']['group_min']
    group_max = constraints['group']['group_max']
    groups = constraints['group']['groups']
    group_vals = list(groups.values())
    weights = np.array(portfolio.weights)
    group_weights = []
    for i in range(0,len(group_vals)):
        group_weights.append(sum(weights[group_vals[i]]))
    plt.plot(group_weights, '^b', linestyle = '--', linewidth= 1.5, c = 'b')
    plt.plot(group_min, '*r', linestyle = ':', c = 'gray')
    plt.plot(group_max, '*r', linestyle = ':', c = 'gray')
    plt.xticks(list(range(0, len(groups))), list(groups.keys()))
    plt.show()


# In[51]:


def chart_efficient_frontier(portfolio, R, metric = 'Sharpe Ratio', arguments = {'rf':0.00}, cml = False,
                   rand_pf = 300, optimize_method = 'DEoptim', alpha = 0.4, figsize = (10,6), **kwargs):
    """
    
    Function to plot efficient frontier of portfolio.
    
    Parameters
    ----------
    portfolio : portfolio_spec,
        an object of class portfolio_spec.
    R : pandas dataframe,
        dataframe of the returns series
    metric: str, default = 'Sharpe Ratio'
        metric should be either of 'Sharpe Ratio' or 'Treynor Ratio'
    arguments : dict, optional
        additional arguments such as risk-free rate in the form of a dictionary.
    cml : bool, default = False
        bool to enable or disable capital market line
    rand_pf : int, optional
        random portfolio to plot to show the efficient frontier
    optimize_method : str, optional
        optimize_method should be similar to those in optimize_portfolio. see optimize_portfolio
    alpha : float, optional
        transparency
    figsize : tuple, optional
        figure size in the form of a tuple
    kwargs : additional key word arguments, optional
        any additional constraint argument to be passed.

    
    Returns
    -------
    matplotlib.scatter
    
    See Also
    --------
    chart_group_weights
    chart_weights
    chart_efficient_frontier
    
    
    Examples
    --------
    >>> # Rb is S&P500 benchmark for capital market line and beta for treynor ratio
    >>> chart_efficient_froniter(portfolio, R, metric = 'Treynor Ratio', 
        arguments = {'rf': 0.0008, 'Rb':Rb}, cml = True, alpha = 0.1, rand_pf = 1000, optimize_method = 'pso')
"""


    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import math as math
    import random






    constraints = get_constraints(portfolio)
    if not (constraints.get('box') == None):
        minimum = constraints['box']['minimum']
        maximum = constraints['box']['maximum']
    if not (constraints.get('long_only') == None):
        minimum = constraints['long_only']['minimum']
        maximum = constraints['long_only']['maximum']

    NP = rand_pf
    init = []
    w = [0]*NP
    for i in range(0,NP):
        sw = []
        for k in range(0, len(portfolio.assets)):
            sw.append(random.uniform(minimum[k], maximum[k]))
        w[i] = sw

    ef_weights = w
    ef_mean = [0]*len(ef_weights)
    ef_sd = [0]*len(ef_weights)

    for i in range(0, len(ef_weights)):
        mu = list(np.mean(R))
        ef_mean[i] = float(port_mean(ef_weights[i], mu))
        ef_sd[i] = float(np.sqrt(var_portfolio(R, ef_weights[i])))

    
    if not (np.all(portfolio.weights) == None):
        opt_w = np.array(portfolio.weights)
    elif np.all(portfolio.weights) == None:
        opt_w = optimize_portfolio(R, portfolio, optimize_method = optimize_method, disp = False, **kwargs)
        opt_w = list(opt_w[0]['weights'].values())

    
    
    mu = list(np.mean(R))

    opt_ret = [port_mean(opt_w, mu)]
    opt_sd = [np.sqrt(var_portfolio(R, opt_w))]

    if metric == 'Sharpe Ratio':
        sd = ef_sd
        if not (arguments.get('rf') == None):
            rf = arguments['rf']
        else:
            rf = 0
        r_premium = np.array(ef_mean)-rf
        tmp_measure = r_premium/sd
    if metric == 'Treynor Ratio':
        tmp_beta = []
        Rb = arguments['Rb']
        for i in range(0,R.shape[1]):
            tmp_beta.append(np.cov(np.transpose(R.iloc[:,i]), Rb)[0,0]/np.var(Rb))
        beta = np.matrix(tmp_beta).dot(np.transpose(np.matrix(ef_weights)))

        if not (arguments.get('rf') == None):
            rf = arguments['rf']
        else:
            rf = 0
        r_premium = np.array(ef_mean)-rf
        tmp_measure = r_premium/beta
        tmp_measure = tmp_measure.tolist()[0]
    elif not (metric == 'Sharpe Ratio' or metric == 'Treynor Ratio'):
        raise ValueError('metric must be either "Sharpe Ratio", or "Treynor Ratio"')

    fig = plt.figure(figsize = figsize)
    plt.scatter(x = ef_sd, y = ef_mean, alpha = 0.4, c = tmp_measure, cmap = 'coolwarm', marker = 'o')
    plt.plot(opt_sd, opt_ret, marker = 'o', markersize = 8, c = 'black')
    plt.text(opt_sd[0], opt_ret[0], 'Optimal', fontsize = 12)
    
    if cml == True:
        if not (np.any(arguments.get('Rb')) == None):
            Rb = arguments['Rb']
        else:
            raise ValueError('Please add benchmark index in arguments')
        tmp_beta = []
        Rb = arguments['Rb']
        for i in range(0,R.shape[1]):
            tmp_beta.append(np.cov(np.transpose(R.iloc[:,i]), Rb)[0,0]/np.var(Rb))
        beta = float(np.matrix(tmp_beta).dot(np.transpose(np.matrix(opt_w))))
        cx = np.linspace(rf, max(ef_mean))
        Ra = rf + beta * np.array(cx)
        plt.plot(cx, Ra)
        if not (arguments.get('rf') == None):
            rf = arguments['rf']
        else:
            rf = 0
    elif cml == False:
        None
        
    plt.title('Efficient Frontier')
    plt.xlabel('Expected Volatility (Std. Dev)')
    plt.ylabel('Expeected Returns')
    plt.colorbar(label = metric)


# In[52]:


def optimize_portfolio(R, portfolio = None, constraints = None, objectives = None, 
                       optimize_method = ['pso', 'DEoptim', 'dual_annealing', 'brute', 'shgo','basinhopping', 'best'], 
                       search_size = 20000, trace = False, message = False, **kwargs):
    
    """
    This function is calls the portfolio to optimize the weights given the constrained and objectives of portfolio
    provided using add_constraint and add_objective functions.
    
    Main function to optimize portfolio weights given the constraints and objectives
    
    Parameters
    ----------
    R : pandas dataframe,
        dataframe of the series of returns
    portfolio : portfolio_spec,
        an object of class portfolio_spec. 
    constraints : dict, optional
        constraints to be minimized/maximized given the assets. 
        Although they are automatically called if portfolio_spec object has constraints specified
    objectives : dict, optional
        objectives to be minimized/maximized given the assets. 
        Although they are automatically called if portfolio_spec object has objectives specified
    optimize_method : float,
        the method for optimizing portfolio. currently supported methods are:
        'pso'
        'DEoptim'
        'dual_annealing'
        'brute'
        'shgo'
        'basinhopping'
        
        User can also specify 'best' as optimize_method to use all optimizer and choose the best among them.
    search_size: int, optional
        specify th iterations to do before calling the best portfolio    
    trace : bool, default = False
        bool to enable or disable trace of portfolio.
    message : bool, default = False
        bool to enable or disable messages.
    kwargs : additional key word arguments, optional
        any additional constraint argument to be passed.
 
   
    
    Returns
    -------
    returns the optimal weights, objective measure, value of the best optimization (out). automatically adds weights
    to the portfolio_spec objects.
    
    See Also
    --------
    portfolio_spec
    add_constraint
    add_objective
    
    Notes
    -----
    currently suuported optimization methods are:
        'pso'
        'DEoptim'
        'dual_annealing'
        'brute'
        'shgo'
        'basinhopping'
        
    The function calls constrained_objective to minimize the penalty (default = 10000) to come to a solution.
    the purpose of adding different optimization method is to give flexibility to user to choose optimizer based on
    specific problem.
    
    Additional arguments can be specified such as maxiter, disp and other controls. the argument names are similar in
    scipy.optimize.[optimizer] or pyswarms. see scipy and pyswarm documentation.

    
    Examples
    --------
    >>> port = portfolio_spec(assets = 5)
    >>> import pandas_datareader as pdr
    >>> aapl = pdr.get_data_yahoo('AAPL')
    >>> msft = pdr.get_data_yahoo('MSFT')
    >>> tsla = pdr.get_data_yahoo('TSLA')
    >>> uber = pdr.get_data_yahoo('UBER')
    >>> amzn = pdr.get_data_yahoo('AMZN')
    >>> port = pd.DataFrame({'aapl': pd.DataFrame.reset_index(aapl).iloc[:,6], 'msft':pd.DataFrame.reset_index(msft).iloc[:,6],
        'tsla': pd.DataFrame.reset_index(tsla).iloc[:,6], 'uber':pd.DataFrame.reset_index(uber).iloc[:,6],
        'amzn': pd.DataFrame.reset_index(amzn).iloc[:,6]})
    >>> port_ret = port.pct_change().dropna()
    >>> R = port_ret
    >>> add_constraint('long_only')
    >>> add_constraint('full_investment')
    >>> #adding objectives
    >>> add_objective(kind = 'return', name = 'mean', target = 0.0018)
    >>> add_objective(kind = 'portfolio_risk', name = 'std', target = 0.015)
    >>> add_objective(kind = 'risk_budget', name = 'risk_budget')
    >>> add_objective(kind = 'weight_conc', name = 'HHI', target = 0.11)
    >>> add_objective(kind = 'performance_metrics', name = 'sharpe', target = 0.13)
    >>> # add a custom objective by first defining it.
    >>> def sortino_ratio(w,R):
            #SOME CODE
    >>> add_objective(kind = 'performance_metrics', name = {'sortino':sortino_ratio}, target = 0.35)
    NOTE: The output of sortino_ratio or other custom function in objective must be a float.
    NOTE: you can also add other custom function in other kind of objective in similar methd.

    >>> optimize_portfolio(R, port, optimize_method  = 'DEoptim', disp = False, search_size = 30000)
    NOTE: Yout can provide addional adguments based on solver. See the solver documentation for more information.
    >>> # we have used scipy.optimize for most of the solver and pyswarms for 'pso' method.
    
    >>> optimize_portfolio(R, port, optimize_method  = 'best', disp = False, maxiter = 1000)
    NOTE: 'best' optimize_method takes a while to give ouput because it uses all other optimizers.
    
    """
    import numpy as np
    import pandas as pd
    import math as math
    import random



    
    if type(portfolio) == list:
        n_porf = len(portfolio)
        out_list = [0]*n_porf
        portfolio_name = ['portfolio']*n_porf
        for i in range(0,n_porf):
            print('starting optimization of: portfolio {}'.format(i))
            out_list[i] = optimize_portfolio(R, portfolio = portfolio[i], constraints = constraints, 
                                            objectives = objectives, optimize_method = optimize_method, 
                                        search_size = search_size, **kwargs)
            portfolio_name[i] = ''.join(['portfolio', str(i)])
        out = dict(zip(portfolio_name, out_list))
        return(out)
    
    optimize_method = optimize_method
    tmpt_trace = None
    if portfolio == None:
        raise ValueError('You must specify a portfolio')
    R = R.dropna()
    N = len(portfolio.assets)
    if len(R.columns)>N:
        R.columns = portfolio.assets
    T = R.shape[0]
    out = dict()
    weights = None
    
    
    #Extra (test required)
    if not (constraints == None) or not (objectives == None):
        portfolio.constraints = constraints
        portfolio.objectives = objectives
        
        
        
    constraints = get_constraints(portfolio = portfolio)
    
    
    def normalize_weights(weights):
        if not (constraints.get('weight_sum') is None):
            if not (constraints.get('weight_sum') is None):
                max_sum = constraints['weight_sum']['max_sum']
                if sum(weights) > max_sum:
                    weights = (max_sum/sum(weights)) * weights
            if not (constraints.get('weight_sum') is None):
                min_sum = constraints['weight_sum']['min_sum']
                if sum(weights) < min_sum:
                    weights = (min_sum/sum(weights)) * weights
        return(weights)
    if optimize_method == 'DEoptim':
        #customize later !!!! add parallel support
        
        
        if not (kwargs.get('strategy') == None):
            strategy = kwargs['strategy']
        else:
            strategy = 'currenttobest1bin'
        
        if not (kwargs.get('maxiter') == None):
            maxiter = kwargs['maxiter']
        else:
            maxiter = N * 50
        
        if not (kwargs.get('disp') == None):
            disp = kwargs['disp']
        else:
            disp = True
            
        if not (kwargs.get('workers') == None):
            workers = kwargs['workers']
        else:
            workers = 1
        
        
        maxiter = N * 50
        NP = round(search_size/maxiter)
        if NP < (N*10):
            NP = N * 10
        if NP > 2000:
            NP = 2000
        if maxiter < 50:
            maxiter = 50
        popsize = NP
        try:
            if not (constraints.get('box') == None):
                minimum = constraints['box']['minimum']
                maximum = constraints['box']['maximum']
            if not (constraints.get('long_only') == None):
                minimum = constraints['long_only']['minimum']
                maximum = constraints['long_only']['maximum']
        except:
            minimum = np.repeat(0, len(portfolio.assets))
            maximum = np.repeat(1, len(portfolio.assets))


        init = []
        w = [0]*NP
        for i in range(0,NP):
            sw = []
            for k in range(0, len(portfolio.assets)):
                sw.append(random.uniform(minimum[k], maximum[k]))
            w[i] = sw

        for i in range(0,NP):
            init.append(fn_map(w[i],portfolio)['weights'])
        from scipy.optimize import differential_evolution, Bounds
        bounds = Bounds(minimum, maximum)
        normalize = False
        arguments = (R, portfolio,)
        try:
            minw = differential_evolution(func = constrained_objective, bounds = bounds, args = arguments, popsize = NP,
                                        init = init, maxiter = maxiter, disp = disp, strategy = strategy, workers = workers)
        except:
            minw = None
       
        if minw == None:
            raise ValueError('optimizer did not find any solution, trying relaxing constraints or checkdata')
        if not (minw == None):
            weights = minw.x
            
            if not (portfolio.objectives == None):
                obj_vals = constrained_objective(w = weights, R = R, portfolio = portfolio, trace = True, storage = True)['objective_measures']
            else:
                obj_vals = 'No Objectives'

            assetnames = portfolio.assets
            out = [{'weights':dict(zip(assetnames, weights))}, {'objective_measures':obj_vals}, {'best':minw.fun}]
                        
            portfolio.weights = weights
            portfolio.objective_measures = obj_vals
            return(out)
    
    if optimize_method == 'pso':
        
        #inputs Update customizations of *args
        if not (kwargs.get('n_particles') == None):
            n_particles = kwargs['n_particles']
        else:
            n_particles = N*50
        
        if not (kwargs.get('options') == None):
            if type(options) != dict:
                raise TypeError('options argument must be a dictionary')
            else:
                options = kwargs['options']
        else:
            options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        
        if not (kwargs.get('iters') == None):
            iters = kwargs['iters']
        else:
            iters = N * 50
        
        if not (kwargs.get('n_processes') == None):
            n_processes = kwargs['n_processes']
        else:
            n_processes = None
        
        
        
        
        
        try:
            if not (constraints.get('box') == None):
                minimum = constraints['box']['minimum']
                maximum = constraints['box']['maximum']
            if not (constraints.get('long_only') == None):
                minimum = constraints['long_only']['minimum']
                maximum = constraints['long_only']['maximum']
        except:
            minimum = np.repeat(0, len(portfolio.assets))
            maximum = np.repeat(1, len(portfolio.assets))
            
            
        x_min = minimum
        x_max = maximum
        bounds = (x_min, x_max)
        
        
        
        
        #Test of init
        NP = n_particles
        init = []
        w = [0]*NP
        for i in range(0,NP):
            sw = []
            for k in range(0, len(portfolio.assets)):
                sw.append(random.uniform(minimum[k], maximum[k]))
            w[i] = sw

        for i in range(0,NP):
            init.append(fn_map(w[i],portfolio)['weights'])
            
        init = np.array(init)
        #Process
        
        from pyswarms.single.global_best import GlobalBestPSO
        
        
        optimizer = GlobalBestPSO(n_particles=n_particles, dimensions=N, options=options, bounds=bounds,
                                 init_pos = init)

        def f(x, R, portfolio):
            n_particles = x.shape[0]
            j = [constrained_objective(x[i], R, portfolio) for i in range(n_particles)]
            return np.array(j)


        #Runn
        try:
            minw = optimizer.optimize(f, iters = iters, R = R, portfolio = portfolio, n_processes = n_processes)
        except:
            minw = None

        if minw == None:
                raise ValueError('optimizer did not find any solution, trying relaxing constraints or checkdata')
        if not (minw == None):
            best = minw[0]
            weights= minw[1]
            weights = normalize_weights(weights)
            
            if not (portfolio.objectives == None):
                obj_vals = constrained_objective(w = weights, R = R, portfolio = portfolio, trace = True, storage = True)['objective_measures']
            else:
                obj_vals = 'No Objectives'

            assetnames = portfolio.assets
            out = [{'weights':dict(zip(assetnames, weights))}, {'objective_measures':obj_vals}, {'best':best}]         
            portfolio.weights = weights
            portfolio.objective_measures = obj_vals
            return(out)
        
        
    if optimize_method == 'dual_annealing':
        
        
        if not (kwargs.get('maxiter') == None):
            maxiter = kwargs['maxiter']
        else:
            maxiter = N * 50
        
        if not (kwargs.get('initial_temp') == None):
            initial_temp = kwargs['initial_temp']
        else:
            initial_temp = 5230
        
        if not (kwargs.get('visit') == None):
            visit = kwargs['visit']
        else:
            visit = 2.62
        
        if not (kwargs.get('accept') == None):
            accept = kwargs['accept']
        else:
            accept = -5.0
        
        if not (kwargs.get('no_local_search') == None):
            no_local_search = kwargs['no_local_search']
        else:
            no_local_search = False
        
        
        
        
        NP = maxiter
        
        
        try:
            if not (constraints.get('box') == None):
                minimum = constraints['box']['minimum']
                maximum = constraints['box']['maximum']
            if not (constraints.get('long_only') == None):
                minimum = constraints['long_only']['minimum']
                maximum = constraints['long_only']['maximum']
        except:
            minimum = np.repeat(0, len(portfolio.assets))
            maximum = np.repeat(1, len(portfolio.assets))


        from scipy.optimize import dual_annealing, Bounds

        x_min = minimum
        x_max = maximum
        bounds = list(zip(x_min, x_max))

        normalize = False
        arguments = (R, portfolio,)

        try:
            minw = dual_annealing(func = constrained_objective, bounds = bounds, args = arguments, 
                                            maxiter = maxiter, initial_temp = initial_temp, 
                                 visit = visit, accept = accept, no_local_search = no_local_search)
        except:
            minw = None



        if minw == None:
            raise ValueError('optimizer did not find any solution, trying relaxing constraints or checkdata')
        if not (minw == None):
            weights = minw.x
            weights = normalize_weights(weights)
            if not (portfolio.objectives == None):
                obj_vals = constrained_objective(w = weights, R = R, portfolio = portfolio, trace = True, storage = True)['objective_measures']
            else:
                obj_vals = 'No Objectives'
            
            assetnames = portfolio.assets
            out = [{'weights':dict(zip(assetnames, weights))}, {'objective_measures':obj_vals}, {'best':minw.fun}]
            
            portfolio.weights = weights
            portfolio.objective_measures = obj_vals
            return(out)
            
            
      #Brute      
            
            
    if optimize_method == 'brute':
        
        if not (kwargs.get('Ns') == None):
            Ns = kwargs['Ns']
        else:
            Ns = 5
        if not (kwargs.get('disp') == None):
            disp = kwargs['disp']
        else:
            disp = True

        if not (kwargs.get('workers') == None):
            workers = kwargs['workers']
        else:
            workers = 1



        try:
            if not (constraints.get('box') == None):
                minimum = constraints['box']['minimum']
                maximum = constraints['box']['maximum']
            if not (constraints.get('long_only') == None):
                minimum = constraints['long_only']['minimum']
                maximum = constraints['long_only']['maximum']
        except:
            minimum = np.repeat(0, len(portfolio.assets))
            maximum = np.repeat(1, len(portfolio.assets))




        from scipy.optimize import brute, Bounds

        x_min = minimum
        x_max = maximum
        bounds = list(zip(x_min, x_max))
        normalize = False
        arguments = (R, portfolio,)


        try:
            minw = brute(func = constrained_objective, ranges = bounds, args = arguments, 
                Ns = Ns, disp = disp, workers= workers, full_output = True)
        except:
            minw = None

        if minw == None:
            raise ValueError('optimizer did not find any solution, trying relaxing constraints or checkdata')
        if not (minw == None):
            weights = minw[0]
            weights = normalize_weights(weights)
            if not (portfolio.objectives == None):
                obj_vals = constrained_objective(w = weights, R = R, portfolio = portfolio, trace = True, storage = True)['objective_measures']
            else:
                obj_vals = 'No Objectives'

            assetnames = portfolio.assets
            out = [{'weights':dict(zip(assetnames, weights))}, {'objective_measures':obj_vals}, {'best':minw[1]}]

            portfolio.weights = weights
            portfolio.objective_measures = obj_vals
            return(out)
    
    
    if optimize_method == 'shgo':
        
        if not (kwargs.get('iters_shgo') == None):
            iters_shgo = kwargs['iters_shgo']
        else:
            iters_shgo = 3
        if not (kwargs.get('sampling_method') == None):
            sampling_method = kwargs['sampling_method']
        else:
            sampling_method = 'simplicial'

        if not (kwargs.get('options') == None):
            if type(kwargs['options']) != dict:
                raise TypeError('options argument must be a dictionary')
            else:
                options = kwargs['options']
        else:
            options = None

        if not (kwargs.get('n') == None) and (kwargs.get('sampling_method') == 'sobol'):
            n = kwargs['n']
        else:
            n = None





        try:
            if not (constraints.get('box') == None):
                minimum = constraints['box']['minimum']
                maximum = constraints['box']['maximum']
            if not (constraints.get('long_only') == None):
                minimum = constraints['long_only']['minimum']
                maximum = constraints['long_only']['maximum']
        except:
            minimum = np.repeat(0, len(portfolio.assets))
            maximum = np.repeat(1, len(portfolio.assets))




        from scipy.optimize import shgo, Bounds

        x_min = minimum
        x_max = maximum
        bounds = list(zip(x_min, x_max))
        normalize = False
        arguments = (R, portfolio,)

        try:
            minw = shgo(func = constrained_objective, bounds = bounds, args = arguments, 
                iters = iters_shgo, sampling_method = sampling_method, n = n)
        except:
            minw = None


        if minw == None:
            raise ValueError('optimizer did not find any solution, trying relaxing constraints or checkdata')
        if not (minw == None):
            weights = minw.x
            weights = normalize_weights(weights)
            if not (portfolio.objectives == None):
                obj_vals = constrained_objective(w = weights, R = R, portfolio = portfolio, trace = True, storage = True)['objective_measures']
            else:
                obj_vals = 'No Objectives'

            assetnames = portfolio.assets
            out = [{'weights':dict(zip(assetnames, weights))}, {'objective_measures':obj_vals}, {'best':minw.fun}]

            portfolio.weights = weights
            portfolio.objective_measures = obj_vals
            return(out)
        
        
    if optimize_method == 'basinhopping':
        
        if not (kwargs.get('niter') == None):
            niter = kwargs['niter']
        else:
            niter = N * 50

        if not (kwargs.get('disp') == None):
            disp = kwargs['disp']
        else:
            disp = True


        try:
            if not (constraints.get('box') == None):
                minimum = constraints['box']['minimum']
                maximum = constraints['box']['maximum']
            if not (constraints.get('long_only') == None):
                minimum = constraints['long_only']['minimum']
                maximum = constraints['long_only']['maximum']
        except:
            minimum = np.repeat(0, len(portfolio.assets))
            maximum = np.repeat(1, len(portfolio.assets))

        NP = 1
        init = []
        w = [0]*NP
        arguments = (R, portfolio,)



        for i in range(0,NP):
            sw = []
            for k in range(0, len(portfolio.assets)):
                sw.append(random.uniform(minimum[k], maximum[k]))
            w[i] = sw

        for i in range(0,NP):
            init.append(fn_map(w[i],portfolio)['weights'])
        from scipy.optimize import basinhopping, Bounds

        def f(x, R, portfolio):
            n_particles = x.shape[0]
            j = [constrained_objective(x[i], R, portfolio) for i in range(n_particles)]
            return np.array(j)




        try:
            minw = basinhopping(func = constrained_objective, x0 = init, niter = niter, minimizer_kwargs = {'args':arguments},
                                        disp = disp)
        except:
            minw = None

        if minw == None:
            raise ValueError('optimizer did not find any solution, trying relaxing constraints or checkdata')
        if not (minw == None):
            weights = minw.x

            if not (portfolio.objectives == None):
                obj_vals = constrained_objective(w = weights, R = R, portfolio = portfolio, trace = True, storage = True)['objective_measures']
            else:
                obj_vals = 'No Objectives'

            assetnames = portfolio.assets
            out = [{'weights':dict(zip(assetnames, weights))}, {'objective_measures':obj_vals}, {'best':minw.fun}]

            portfolio.weights = weights
            portfolio.objective_measures = obj_vals
            return(out)    
        
        

    if optimize_method == 'best':
        
        if not (kwargs.get('strategy') == None):
            strategy = kwargs['strategy']
        else:
            strategy = 'currenttobest1bin'
        
        if not (kwargs.get('maxiter') == None):
            maxiter = kwargs['maxiter']
        else:
            maxiter = N * 50
        
        if not (kwargs.get('disp') == None):
            disp = kwargs['disp']
        else:
            disp = True
            
        if not (kwargs.get('workers') == None):
            workers = kwargs['workers']
        else:
            workers = 1
        
        #inputs Update customizations of *args
        if not (kwargs.get('n_particles') == None):
            n_particles = kwargs['n_particles']
        else:
            n_particles = 100
        
        if not (kwargs.get('options') == None):
            options = kwargs['options']
        else:
            options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        
        if not (kwargs.get('iters') == None):
            iters = kwargs['iters']
        else:
            iters = N * 50
        
        if not (kwargs.get('n_processes') == None):
            n_processes = kwargs['n_processes']
        else:
            n_processes = None

      
        if not (kwargs.get('initial_temp') == None):
            initial_temp = kwargs['initial_temp']
        else:
            initial_temp = 5230
        
        if not (kwargs.get('visit') == None):
            visit = kwargs['visit']
        else:
            visit = 2.62
        
        if not (kwargs.get('accept') == None):
            accept = kwargs['accept']
        else:
            accept = -5.0
        
        if not (kwargs.get('no_local_search') == None):
            no_local_search = kwargs['no_local_search']
        else:
            no_local_search = False


        if not (kwargs.get('Ns') == None):
            Ns = kwargs['Ns']
        else:
            Ns = 5
       
        if not (kwargs.get('iters_shgo') == None):
            iters_shgo = kwargs['iters_shgo']
        else:
            iters_shgo = 3
        if not (kwargs.get('sampling_method') == None):
            sampling_method = kwargs['sampling_method']
        else:
            sampling_method = 'simplicial'

        if not (kwargs.get('options') == None):
            if type(kwargs['options']) != dict:
                raise TypeError('options argument must be a dictionary')
            else:
                options = kwargs['options']
        else:
            options = None

        if not (kwargs.get('n') == None) and (kwargs.get('sampling_method') == 'sobol'):
            n = kwargs['n']
        else:
            n = None
        
        if not (kwargs.get('niter') == None):
            niter = kwargs['niter']
        else:
            niter = N * 50



        
        print('This may take a will as it will optimize weights sequentially across all optimizers')
        
        optimizer = ['DEoptim', 'pso', 'dual_annealing', 'brute', 'shgo', 'basinhopping']
        tmp_out = []
        for optimize in optimizer:
            tmp_out.append(optimize_portfolio(R, portfolio, optimize_method = optimize, **kwargs))
        best_vals = []
        for i in range(0,len(tmp_out)):
            best_vals.append(list(tmp_out[i][2].values()))
        tmp_out = np.array(tmp_out)
        out = tmp_out[np.where(min(best_vals))][0]
            
            
        if np.all(out) == None:
            raise ValueError('optimizer did not find any solution, trying relaxing constraints or check data')
        if not (np.all(out) == None):
            print('best optimizer was: {}'.format(np.array(optimizer)[np.where(min(best_vals))]))
            return(list(out))
            portfolio.weights = out['weights'] 
            portfolio.objecitve_measures = out['objective_measures']


# In[53]:


def equal_weight(R, portfoio, **kwargs):
    
    """
    
    Function to extract objective measures of equal weight portfolio.
    
    Parameters
    ----------
    R : pd.DataFrame,
        dataframe of returns of assets in portfolio
    portfolio : portfolio_spec,
        an object of class portfolio_spec.  
    kwargs : additional arguments, optional
        any additional argument to be passed.
    
    Returns
    -------
    dictionary of objective measures of equal weighted portfolio
    
    See Also
    --------
    inverse_volatility_weights 
    
    Examples
    --------
    >>> equal_weight(R, portfolio)
    
    """

    import numpy as np
    import pandas as pd

    import math as math
    import random




    assets = portfolio.assets
    nassets = len(assets)
    weights = np.repeat(1/nassets, nassets)
    if nassets != R.shape[1]:
        raise ValueError('assets should be of same length as columns in R')
    tmp_out = constrained_objective(weights, R, portfolio, trace = True)
    return(tmp_out)


# In[54]:


def inverse_volatility_weights(R, portfoio, **kwargs):
    
    """
    
    Function to calculate objective_measure of inverse volatility portfolio
    
    Parameters
    ----------
    R : pd.DataFrame,
        dataframe of returns of assets in portfolio
    portfolio : portfolio_spec,
        an object of class portfolio_spec.  
    kwargs : additional arguments, optional
        any additional argument to be passed.
    
    Returns
    -------
    dictionary of objective measures of inverse volatility portfolio
    
    See Also
    --------
    equal_weight
    inverse_volatility_weight 
    
    Examples
    --------
    >>> inverse_volatility_weight(R, portfolio)
    
    """
    import numpy as np
    import pandas as pd
    import math as math
    import random




    assets = portfolio.assets
    nassets = len(assets)
    weights = np.array((1/np.std(R))/sum(1/np.std(R)))
    if nassets != R.shape[1]:
        raise ValueError('assets should be of same length as columns in R')
    tmp_out = constrained_objective(weights, R, portfolio, trace = True)
    return(tmp_out)

