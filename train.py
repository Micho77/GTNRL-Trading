# Import relevant modules
from agent.agent_DDQNN import Agent
from helpers import getStockDataVec, getState
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import random

# Set random seed
random.seed(0)
np.random.seed(0)

# Get data
file_name = 'g10_daily_carry_rs_2000_2019.csv'
rs_data = pd.read_csv(f'data/{file_name}', index_col=0, header=0)

# Initialize Agent variables
trading_currency = 'EURUSD'
window_size = 252
episode_count = 10
batch_size = 64  # batch size for replaying/training the agent
agent = Agent(state_size=(window_size, rs_data.shape[1]))

# Initialize training variables
total_rewards_across_episodes = []

# Training over episodes
for e in range(episode_count):

    # Print progress
    print(f"Episode: {e+1}/{episode_count}")
    print(f"Epsilon: {agent.epsilon}")

    # Reset agent parameters to run next episode
    agent.episode_reset()

    # Loop over time
    for t in rs_data.index[window_size:window_size+1000]:

        # past {window_size} log returns up to and excluding {t}
        X = rs_data.loc[:t].iloc[-window_size-1:-1]  # fetch raw data
        X = X.values.reshape([1]+list(X.shape))  # tensorize

        # Get action from agent
        action = agent.act(X)

        # Process returns/rewards
        action_direction = -1*(action*2-1) # map 0->buy->+1, 1->sell->-1
        reward = 100*action_direction*rs_data.loc[t, trading_currency]
        agent.reward_episode += reward
        print(agent.reward_episode, reward)

        # Fetch next state
        next_X = rs_data.loc[:t].iloc[-window_size:]  # fetch raw data
        next_X = next_X.values.reshape([1]+list(next_X.shape))  # tensorize
        done = True if t == rs_data.index[-1] else False

        # Append to memory & train
        agent.memory.append((X[0], action, reward, next_X[0], done))
        agent.replay(min(batch_size, len(agent.memory)))

        # Print if done
        if done:
            print("--------------------------------")
            print(f"Episode reward:{agent.reward_episode}%")
            print("--------------------------------")

    # Record episode data
    total_rewards_across_episodes.append(agent.reward_episode)

    # Update agent parameters
    agent.update_target_model()
    agent.epsilon = max(agent.epsilon_decay*agent.epsilon, agent.epsilon_min)

    # # Save models
    # agent.model.save("models/model_ep" + str(e))
    # agent.target_model.save("models/model_target_ep"+str(e))
        
# plt.figure()
# plt.plot(portfolio_values)
# plt.title('Portfolio Values across Episodes')
# plt.savefig('Portvalues.png')
# plt.figure()
# plt.plot(total_rewards_across_episodes)
# plt.title('Reward Evolution across Episodes')
# plt.savefig('Rewards.png')


