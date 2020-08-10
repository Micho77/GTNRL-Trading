from agent.agent_DDQN import Agent
from helpers import getStockDataVec, getState
from  matplotlib import pyplot as plt
import numpy as np
import random

# Set random seed
random.seed(0)
np.random.seed(0)

# Get data & initialize RL variables
stock_name, window_size, episode_count = "usdcad_jan_march_2020", 24, 25
balance = 0
agent = Agent(window_size+2, balance)
data = getStockDataVec(stock_name)

l = len(data) - 1
batch_size = 64

portfolio_values = []
total_rewards_across_episodes = []

# Training over episodes
for e in range(episode_count + 1):

    print("Episode " + str(e) + "/" + str(episode_count))
    print("Epsilon:{}".format(agent.epsilon))

    # Reset agent parameters to run next episode
    agent.buy_inventory = []
    agent.sell_inventory = []
    agent.balance = balance
    reward_episode = 0
    locked_pnl = 0 # overall pnl
    
    # Start with a buy
    actions = []
    actions.append(0) 
    agent.balance -= data[0]
    agent.buy_inventory.append(data[0])
    
    # Initial State at t=1
    state = getState(data, 1, window_size + 1)
    state= np.concatenate((state, np.array(len(agent.sell_inventory)).reshape(1,),
                           np.array(len(agent.buy_inventory)).reshape(1,)),axis=0).reshape(1,window_size+2)
    
    for t in range(1,l):
        
        prev_port_value=(len(agent.buy_inventory)-len(agent.sell_inventory))*data[t]+agent.balance
        
        action = agent.act(state)
        # signals = agent.model.predict(state)[0]
        # print('Hold:{}, Buy:{}, Sell:{}'.format(signals[0],signals[1],signals[2]))
        
        if action == 0 : # Buy signal
            if(actions[t-1]==0):  # If the previous signal was a buy too, no act
                pass
            else:
                ### Pnl 
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
        reward_episode+=reward
        # print(reward)

        done = True if t == l - 1 else False
        next_state = getState(data, t + 1, window_size + 1)
        next_state = np.concatenate((next_state, np.array(len(agent.sell_inventory)).reshape(1,),
                                     np.array(len(agent.buy_inventory)).reshape(1,)),axis=0).reshape(1,window_size+2)
        
        agent.memory.append((state, action, reward, next_state, done))
        
        state = next_state
        actions.append(action)

        agent.replay(min(batch_size,len(agent.memory)))

        if done:
            final_portfolio_value=(len(agent.buy_inventory)-len(agent.sell_inventory))*data[t]+agent.balance
            print("--------------------------------")
            print("Total Portfolio Value:{}".format(final_portfolio_value))
            print("Locked Pnl:{}".format(locked_pnl))
            print("Balance:{}".format(agent.balance))
            print("Mark to Market:{}".format((len(agent.buy_inventory)-len(agent.sell_inventory))*data[t]))
            print("Episode reward:{}".format(reward_episode))
            print("Number of holding currency:{}".format(len(agent.buy_inventory)))
            print("Number of shorted currency:{}".format(len(agent.sell_inventory)))
            print("--------------------------------")
    
    total_rewards_across_episodes.append(reward_episode)
    portfolio_values.append(final_portfolio_value)
    agent.update_target_model()

    if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

    agent.model.save("models/model_ep" + str(e))
    agent.target_model.save("models/model_target_ep"+str(e))
        
plt.figure()
plt.plot(portfolio_values)
plt.title('Portfolio Values across Episodes')
plt.savefig('Portvalues.png')
plt.figure()
plt.plot(total_rewards_across_episodes)
plt.title('Reward Evolution across Episodes')
plt.savefig('Rewards.png')


