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


from .pyPortfolioAnalysis import *
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.optimize
from pyswarms.single.global_best import GlobalBestPSO
from .__version__ import __version__
