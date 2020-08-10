import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

def getStockDataVec(key):
    vec = []
    lines = open("data/" + key + ".csv", "r").read().splitlines()

    for line in lines[1:]:
        vec.append(float(line.split(",")[4])) # 1 for sinusoid/noise

    return vec

def getState(data, t, n):
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0, t+1 includes the prices of time t due to python indexing, NOT a look-ahead bias
    return 10000*np.diff(np.array(block)) 

def plotAction(stock_name, buy_data, sell_data, model_name):
    stock_data = pd.read_csv("data/" + stock_name + ".csv")
    buy_data = np.array(buy_data)
    sell_data = np.array(sell_data)
    l = len(stock_data['Close']) 
    t = np.arange(0,l,1)

    fig, ax = plt.subplots(figsize=(40,20))
    ax.plot(t, stock_data['Close'], label=str(stock_name), color = 'grey') 
    ax.plot(buy_data[:, 0], buy_data[:, 1],'go', markersize = 4.5, label='Buy')
    ax.plot(sell_data[:, 0], sell_data[:, 1], 'ro', markersize = 4.5, label='Sell')
    ax.legend(loc='upper left')
    
    plt.savefig('plots/'+str(stock_name)+'_'+str(model_name)+'.png', dpi=200)
    
def maximum_drawdown(portfolio_values):
    end_index = np.argmax(np.maximum.accumulate(portfolio_values) - portfolio_values)
    if end_index == 0:
        return 0
    beginning_index = np.argmax(portfolio_values[:end_index])
    return (portfolio_values[end_index] - portfolio_values[beginning_index]) / portfolio_values[beginning_index]

def sharpe(portfolio_values):
    hourly_returns=np.diff(portfolio_values)
    return np.mean(hourly_returns)/np.std(hourly_returns)
        

