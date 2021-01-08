# pyPortfolioAnalysis

pyPortfolioAnalysis is a Python library for numeric method for portfolio optimisation.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install pyPortfolioAnalysis
```

## Usage

```python
import pyPortfolioAnalysis
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
p1 = portfolio_spec(assets = ['AAPL', 'MSFT', 'TSLA', 'UBER', 'AMZN'])
add_constraint(p1, 'long_only')
add_constraint(p1, 'full_investment')
add_objective(p1, kind='return', name = 'mean', target = 0.002)
add_objective(p1, kind='risk', name = 'std', target = .018)
p1.port_summary()
constraints = get_constraints(p1)
p1.port_summary()

optimize_portfolio(R, p1, optimize_method = 'DEoptim', disp = False)
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[GPL3](https://choosealicense.com/licenses/gpl-3.0/)
