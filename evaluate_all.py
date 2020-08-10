from keras.models import load_model
from helpers import getStockDataVec, getState, sharpe
import matplotlib.pyplot as plt
from agent.agent_DDQN import Agent
import numpy as np

portfolio_values=[]
rewards=[]
# sharpes=[]

for m in range(0,16):
    
    stock_name, model_name, model_target_name = "usdcad_2019", "model_ep"+str(m), "model_target_ep"+str(m)
    model = load_model("models/" + model_name)
    window_size = 24
    balance=0
    agent = Agent(window_size+2, balance, True, model_name, model_target_name)
    
    data = getStockDataVec(stock_name)
    l = len(data) - 1
    
    plt_data_buy = []
    plt_data_sell = []
    
    total_reward=0
    locked_pnl = 0
    
    # Start with a buy
    actions=[]
    actions.append(0) 
    agent.balance-=data[0]
    agent.buy_inventory.append(data[0])
    
    
    state = getState(data, 1, window_size + 1)
    state= np.concatenate((state, np.array(len(agent.sell_inventory)).reshape(1,), np.array(len(agent.buy_inventory)).reshape(1,)),axis=0).reshape(1,window_size+2)
  
     
    for t in range(1,l):
        
        prev_port_value=(len(agent.buy_inventory)-len(agent.sell_inventory))*data[t]+agent.balance

        action = agent.act(state)
        # signals = agent.model.predict(state)[0]
        # print('Hold:{}, Buy:{}, Sell:{}'.format(signals[0],signals[1],signals[2]))
        
        # Hold
        next_state = getState(data, t + 1, window_size + 1)
        next_state = np.concatenate((next_state, np.array(len(agent.sell_inventory)).reshape(1,), np.array(len(agent.buy_inventory)).reshape(1,)),axis=0).reshape(1,window_size+2)
                
        
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

        total_reward+=reward
        done = True if t == l-1 else False
        state = next_state
        actions.append(action)
     
        if done:
            mark_to_market=(len(agent.buy_inventory)-len(agent.sell_inventory))*data[t]
            current_portfolio_value=agent.balance+mark_to_market
            portfolio_values.append(current_portfolio_value)
            rewards.append(total_reward)
            # sharpes.append(np.sqrt(24*252)*sharpe(portfolio_values)) # need to store portfolio values

            print("Ep:{} done".format(m))
            print("Portfolio value :{}".format(current_portfolio_value))
            print("Number of holding currency:{}".format(len(agent.buy_inventory)))
            print("Number of shorted currency:{}".format(len(agent.sell_inventory)))
            # print("Sharpe Ratio:{}".format(np.sqrt(24*252)*sharpe(portfolio_values)))

plt.figure()
plt.plot(portfolio_values)
plt.title('Portfolio Value across episodes')
plt.savefig('Ports_test.png')


plt.figure()
plt.plot(rewards)
plt.title('Rewards across episodes')

    
