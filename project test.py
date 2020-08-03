from pandas_datareader import data as web
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

assets = ["FB", "AMZN", "AAPL", "NFLX", "GOOG"]
#한국주식 종목코드 문제 005900.KS 아니면 KQ를 추가해야함
#검색 포맷팅 하기 / 미국주식이랑 한국주식 섞여 있을 경우 숫자 통일
# 위에서 찍은 종목코드로 배당금 웹스크레핑 후 차트만들기

weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
weights

stockStartDate = '2019-07-25'
today = datetime.today().strftime('%Y-%m-%d')

df = pd.DataFrame()
for stock in assets:
    df[stock] = web.DataReader(stock,data_source='yahoo',start=stockStartDate , end=today)['Adj Close']
print(df)

title = 'Portfolio Adj. Close Price History'
my_stocks = df
plt.figure(figsize=(12.2,4.5))
for c in my_stocks.columns.values:
  plt.plot(my_stocks[c],  label=c)
plt.title(title)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Adj. Price USD ($)',fontsize=18)
plt.legend(my_stocks.columns.values, loc='upper left')
plt.show()

returns = df.pct_change()
#print(returns)

cov_matrix_annual = returns.cov() * 252
#print(cov_matrix_annual)
port_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
#print(port_variance)
port_volatility = np.sqrt(port_variance)
#print(port_volatility)
portfolioSimpleAnnualReturn = np.sum(returns.mean()*weights) * 252
#print(portfolioSimpleAnnualReturn)
percent_var = str(round(port_variance, 2) * 100) + '%'
percent_vols = str(round(port_volatility, 2) * 100) + '%'
percent_ret = str(round(portfolioSimpleAnnualReturn, 2)*100)+'%'
print("Expected annual return : "+ percent_ret)
print('Annual volatility/standard deviation/risk : '+percent_vols)
print('Annual variance : '+percent_var)

#pip install PyPortfolioOpt
#from pypfopt.efficient_frontier import EfficientFrontier
#from pypfopt import risk_models
#from pypfopt import expected_returns

number_of_portfolios = 3000
RF = 0

portfolio_returns = []
portfolio_risk = []
sharpe_ratio_port = []
portfolio_weights = []

for portfolio in range(number_of_portfolios):
    random_weights = np.random.random_sample((len(assets)))
    random_weights = random_weights / np.sum(random_weights)
    random_port_return = np.sum((returns.mean() * random_weights) * 252)
    portfolio_returns.append(random_port_return)

    cov_matrix_annual = returns.cov() * 252
    random_variance = np.dot(random_weights.T, np.dot(cov_matrix_annual, random_weights))
    random_volatility = np.sqrt(random_variance)
    portfolio_risk.append(random_volatility)

    sharpe_ratio = ((random_port_return - RF)/random_volatility)
    sharpe_ratio_port.append(sharpe_ratio)

    portfolio_weights.append(random_weights)

portfolio_risk = np.array(portfolio_risk)
portfolio_returns = np.array(portfolio_returns)
sharpe_ratio_port = np.array(sharpe_ratio_port)

plt.figure(figsize=(10, 5))
plt.scatter(portfolio_risk, portfolio_returns, c=portfolio_returns / portfolio_risk)
plt.xlabel('volatility')
plt.ylabel('returns')
plt.colorbar(label='Sharpe ratio')

porfolio_metrics = [portfolio_returns,portfolio_risk,sharpe_ratio_port, portfolio_weights]

portfolio_dfs = pd.DataFrame(porfolio_metrics)
portfolio_dfs = portfolio_dfs.T
portfolio_dfs.columns = ['Port Returns','Port Risk','Sharpe Ratio','Portfolio Weights']

for col in ['Port Returns', 'Port Risk', 'Sharpe Ratio']:
    portfolio_dfs[col] = portfolio_dfs[col].astype(float)

#portfolio with the highest Sharpe Ratio
Highest_sharpe_port = portfolio_dfs.iloc[portfolio_dfs['Sharpe Ratio'].idxmax()]
#portfolio with the minimum risk
min_risk = portfolio_dfs.iloc[portfolio_dfs['Port Risk'].idxmin()]

#Highest_sharpe_port
print(Highest_sharpe_port)
print(min_risk)