# pyPortfolioAnalysis

pyPortfolioAnalysis is a Python library for numeric method for portfolio optimisation.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install [pyPortfolioAnalysis](https://pypi.org/project/pyPortfolioAnalysis/).

Documentation is available as docstring or as HTML on https://github.com/anuragagrawaal/pyPortfolioAnalysis/blob/main/pyPortfolioAnalysis.html



```bash
pip install pyPortfolioAnalysis
```

## Usage

```python
from pyPortfolioAnalysis import *
import pandas as pd
#Sample portfolio optimisation
import pandas_datareader as pdr
aapl = pdr.get_data_yahoo('AAPL')
msft = pdr.get_data_yahoo('MSFT')
tsla = pdr.get_data_yahoo('TSLA')
uber = pdr.get_data_yahoo('UBER')
amzn = pdr.get_data_yahoo('AMZN')
port = pd.DataFrame({'aapl': pd.DataFrame.reset_index(aapl).iloc[:,6], 'msft':pd.DataFrame.reset_index(msft).iloc[:,6],
                   'tsla': pd.DataFrame.reset_index(tsla).iloc[:,6], 'uber': pd.DataFrame.reset_index(uber).iloc[:,6],
                    'amzn': pd.DataFrame.reset_index(amzn).iloc[:,6]})
port_ret = port.pct_change().dropna()
p1 = portfolio_spec(assets = ['AAPL', 'MSFT', 'TSLA', 'UBER', 'AMZN'])
add_constraint(p1, 'long_only')
add_constraint(p1, 'full_investment')
add_objective(p1, kind='return', name = 'mean', target = 0.002)
add_objective(p1, kind='risk', name = 'std', target = .018)
p1.port_summary()
constraints = get_constraints(p1)
p1.port_summary()

optimize_portfolio(port_ret, p1, optimize_method = 'DEoptim', disp = False)
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Authors
Anurag Agrawal

[![paypal](https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif)](paypal.me/anuragagrawal1)

## License
[GPL3](https://choosealicense.com/licenses/gpl-3.0/)

## References
Brian G. Peterson and Peter Carl (2018). PortfolioAnalytics: Portfolio Analysis, Including Numerical Methods for Optimization of Portfolios. R package version 1.1.0. https://CRAN.R-project.org/package=PortfolioAnalytics

Boudt, Kris and Lu, Wanbo and Peeters, Benedict, Higher Order Comoments of Multifactor Models and Asset Allocation (June 16, 2014). Available at SSRN: http://ssrn.com/abstract=2409603 or http://dx.doi.org/10.2139/ssrn.2409603

Chriss, Neil A and Almgren, Robert, Portfolios from Sorts (April 27, 2005). Available at SSRN: http://ssrn.com/abstract=720041 or http://dx.doi.org/10.2139/ssrn.720041

Meucci, Attilio, The Black-Litterman Approach: Original Model and Extensions (August 1, 2008). Shorter version in, THE ENCYCLOPEDIA OF QUANTITATIVE FINANCE, Wiley, 2010. Avail- able at SSRN: http://ssrn.com/abstract=1117574 or http://dx.doi.org/10.2139/ssrn.1117574

Meucci, Attilio, Fully Flexible Views: Theory and Practice (August 8, 2008). Fully Flexible Views: Theory and Practice, Risk, Vol. 21, No. 10, pp. 97-102, October 2008. Available at SSRN: http://ssrn.com/abstract=1213325

Scherer, Bernd and Martin, Doug, Modern Portfolio Optimization. Springer. 2005.

Shaw, William Thornton, Portfolio Optimization for VAR, CVaR, Omega and Utility with General Return Distributions: A Monte Carlo Approach for Long-Only and Bounded Short Portfolios with Optional Robustness and a Simplified Approach to Covariance Matching (June 1, 2011). Available at SSRN: http://ssrn.com/abstract=1856476 or http://dx.doi.org/10.2139/ssrn.1856476

