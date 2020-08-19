from flask import Flask, render_template, jsonify, request
from pymongo import MongoClient
from pandas_datareader import data as web
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

app = Flask(__name__)

client = MongoClient('localhost', 27017)  # mongoDB는 27017 포트로 돌아갑니다.
db = client.dbsparta  # 'dbsparta'라는 이름의 db를 만들거나 사용합니다.

@app.route('/')
def home():
    return render_template('ProjectHTML.html')

@app.route('/portfolio', methods=['POST'])
def calculation():
    ticker1 = request.form['ticker1']
    ticker2 = request.form['ticker2']
    weight1 = request.form['weight1']
    weight2 = request.form['weight2']

    assets = ['ticker1', 'ticker2']
    #한국주식 종목코드 문제 005900.KS 아니면 KQ를 추가해야함
    #검색 포맷팅 하기 / 미국주식이랑 한국주식 섞여 있을 경우 숫자 통일
    # 위에서 찍은 종목코드로 배당금 웹스크레핑 후 차트만들기

    weights = np.array([weight1, weight2])

    stockStartDate = '2019-07-25'
    today = datetime.today().strftime('%Y-%m-%d')

    df = pd.DataFrame()
    for stock in assets:
        df[stock] = web.DataReader(stock, data_source='yahoo', start=stockStartDate, end=today)['Adj Close']

    my_stocks = df
    title = 'Portfolio Adj. Close Price History'
    #### 차트 제목에 한글폰트가 없어서 ㅁ 로 나오는 현상 발생
    #### 하단 좌측 usd부분도 마찬가지
    plt.figure(figsize=(12.2,4.5))
    for c in my_stocks.columns.values:
      plt.plot(my_stocks[c],  label=c)
    plt.title(title)
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Adj. Price USD ($)',fontsize=18)
    plt.legend(my_stocks.columns.values, loc='upper left')
    plt.show()

    returns = df.pct_change()
    cov_annual = returns.cov() * 252
    port_variance = np.dot(weights.T, np.dot(cov_annual, weights))
    port_volatility = np.sqrt(port_variance)
    port_returns = np.sum(returns.mean()*weights) * 252

    percent_var = str(round(port_variance, 2) * 100) + '%'
    percent_vol = str(round(port_volatility, 2) * 100) + '%'
    percent_ret = str(round(port_returns, 2)*100)+'%'

    print("포트폴리오 연간 수익률 : "+ percent_ret)
    print('포트폴리오 연간 변동성/리스크 : '+percent_vol)
    print('포트폴리오 연간 분산 : '+percent_var)


    ### random weight 생성 후 무한 반복 계산하여 새로운 리턴,분산 계산하기

    number_of_portfolios = 5000
    RiskFree = 0

    portfolio_returns = []
    portfolio_risk = []
    sharpe_ratio_port = []
    portfolio_weights = []

    for portfolio in range(number_of_portfolios):
        weights = np.random.random_sample((len(assets)))
        weights = weights / np.sum(weights)
        random_port_return = np.sum((returns.mean() * weights) * 252)
        portfolio_returns.append(random_port_return)

        random_variance = np.dot(weights.T, np.dot(cov_annual, weights))
        random_volatility = np.sqrt(random_variance)
        portfolio_risk.append(random_volatility)

        sharpe_ratio = ((random_port_return - RiskFree)/random_volatility)
        sharpe_ratio_port.append(sharpe_ratio)

        portfolio_weights.append(weights)

    ### 차트 그리기
    portfolio_risk = np.array(portfolio_risk)
    portfolio_returns = np.array(portfolio_returns)
    sharpe_ratio_port = np.array(sharpe_ratio_port)

    plt.figure(figsize=(10, 5))
    plt.scatter(portfolio_risk, portfolio_returns, c=portfolio_returns / portfolio_risk)
    plt.xlabel('volatility')
    plt.ylabel('returns')
    plt.colorbar(label='Sharpe ratio')

    ##
    portfolio_metrics = [portfolio_returns, portfolio_risk, sharpe_ratio_port, portfolio_weights]

    portfolio_dfs = pd.DataFrame(portfolio_metrics)
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

    return jsonify({'result': 'success'})

if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)