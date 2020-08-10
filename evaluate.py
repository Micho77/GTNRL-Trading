from keras.models import load_model
from helpers import getStockDataVec, getState, plotAction, maximum_drawdown,sharpe
import matplotlib.pyplot as plt
from agent.agent_DDQN import Agent
import numpy as np
import random

random.seed(0)
np.random.seed(0)

# stock_name, model_name, model_target_name = "usdcad_may_2019", "model_ep14", "model_target_ep14"
# stock_name, model_name, model_target_name = "usdcad_nov_dec_2019", "model_ep10", "model_target_ep10"
stock_name, model_name, model_target_name = "usdcad_may_2019", "model_ep2", "model_target_ep2"

# batch_size=64
model = load_model("models/" + model_name)
window_size = 24
balance=0
agent = Agent(window_size+2, balance, True, model_name, model_target_name)
consecutive_holds=0

data = getStockDataVec(stock_name)
l = len(data) - 1

# Plot 
plt_data_buy = []
plt_data_sell = []

# Statistics
portfolio_values=[]
locked_pnl = 0
locked_pnl_evolution=[]
net_position_evolution=[]
agent_balance_evolution=[]

# Start with a buy
actions=[]
actions.append(0) 
agent.balance-=data[0]
agent.buy_inventory.append(data[0])
plt_data_buy.append((0, data[0]))

state = getState(data, 1, window_size + 1)
state= np.concatenate((state, np.array(len(agent.sell_inventory)).reshape(1,), np.array(len(agent.buy_inventory)).reshape(1,)),axis=0).reshape(1,window_size+2)

for t in range(1,l):
        
    action = agent.act(state)
    # signals = agent.model.predict(state)[0]
    # print('Hold:{}, Buy:{}, Sell:{}'.format(signals[0],signals[1],signals[2]))
    
    next_state = getState(data, t + 1, window_size + 1)
    next_state = np.concatenate((next_state, np.array(len(agent.sell_inventory)).reshape(1,), np.array(len(agent.buy_inventory)).reshape(1,)),axis=0).reshape(1,window_size+2)
        
    prev_port_value=(len(agent.buy_inventory)-len(agent.sell_inventory))*data[t]+agent.balance
    
    if action == 0 : # Buy signal
        if(actions[t-1]==0):  # If the previous signal was a buy too, no act
            pass
        else:
            ### Pnl 
            plt_data_buy.append((t, data[t]))
            sold_price = agent.sell_inventory.pop(0) 
            pnl=sold_price - data[t]
            # print("Buy:{}".format(data[t]) + " | Pnl from closing short position:{}".format(pnl))
            locked_pnl+=pnl
            agent.balance-=data[t]
            ### Taking long position 
            agent.balance-=data[t]
            agent.buy_inventory.append(data[t])
            # print("Buy:{}".format(data[t]))
            
    else: # Sell signal 
        if(actions[t-1]==1): # If the previous signal was a sell too, no act
            pass
        else:
            plt_data_sell.append((t, data[t]))

            ### Pnl 
            bought_price = agent.buy_inventory.pop(0) 
            pnl=data[t]-bought_price
            # print("Sell:{}".format(data[t]) + " | Pnl from closing long position:{}".format(pnl))
            locked_pnl+=pnl
            agent.balance+=data[t]
            ### Taking short position  
            agent.balance+=data[t]
            agent.sell_inventory.append(data[t])
            # print("Sell:{}".format(data[t]))
            
    current_port_value=(len(agent.buy_inventory)-len(agent.sell_inventory))*data[t+1]+agent.balance
    reward=10000*(current_port_value-prev_port_value)

    done = True if t == l - 1 else False
    agent.memory.append((state, action, reward, next_state, done))
    state = next_state
    actions.append(action)

    ##### Monitor portfolio value fluctuations #####
    mark_to_market=(len(agent.buy_inventory)-len(agent.sell_inventory))*data[t]
    current_portfolio_value=agent.balance+mark_to_market
    locked_pnl_evolution.append(locked_pnl)
    net_position_evolution.append(len(agent.buy_inventory)-len(agent.sell_inventory))
    agent_balance_evolution.append(agent.balance)

    portfolio_values.append(current_portfolio_value)
    
    # agent.memory.append((state, action, reward, next_state, done))
    # agent.replay(min(batch_size,len(agent.memory)))
       
    ##### Print final statistics #####
    if done:
        print ("--------------------------------")
        # print("Number of holding currency:{}".format(len(agent.buy_inventory)))
        # print("Number of shorted currency:{}".format(len(agent.sell_inventory)))
        # print("Value of holding currency:{}".format(len(agent.buy_inventory)*data[t]))
        # print("Debt due to short position:{}:".format(-len(agent.sell_inventory)*data[t]))
        # print("Mark to market:{}".format(mark_to_market))
        print("Locked pnl:{}".format(locked_pnl))
        print("Agent balance:{}".format(agent.balance))
        print("Maximum Drawdown:{} %".format(100*maximum_drawdown(portfolio_values)))
        print("Sharpe Ratio:{}".format(np.sqrt(24*252)*sharpe(portfolio_values)))
        print("Final Portfolio Value:{}".format(agent.balance+mark_to_market))
        print ("--------------------------------")

plotAction(stock_name, plt_data_buy, plt_data_sell, model_name)
   
plt.figure()
plt.plot(portfolio_values)
plt.title('Portfolio Value across time')

plt.figure()
plt.plot(locked_pnl_evolution)
plt.title('Pnl across time')

plt.figure()
plt.plot(net_position_evolution)
plt.title('Net Position Evolution across time')

plt.figure()
plt.plot(agent_balance_evolution)
plt.title('Balance across time')
