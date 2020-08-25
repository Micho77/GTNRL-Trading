# Import relevant modules
from agent.agent_DDQNN import RNNAgent, GTNAgent
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import random
import os

# Set random seed
random.seed(0)
np.random.seed(0)

# Initialize Agent variables
trading_currency = 'EURUSD'
window_size = 22
episode_count = 10
batch_size = 64  # batch size for replaying/training the agent

# Initialize training variables
total_rewards_df = pd.DataFrame(dtype=float)

# Get returns data
rs_types = ['carry', 'open', 'high', 'low', 'last']
file_names = [f'g10_daily_{t}_rs_2000_2019.csv' for t in rs_types]
rs_data = dict(zip(rs_types, [pd.read_csv(f'data/{f}', index_col=0, header=0) for f in file_names]))
rs_y = rs_data['carry'][trading_currency]

# Get graphs data
A_t = pd.read_csv('data/A_t_22.csv', index_col=0, header=0)
A_n = pd.read_csv('data/g10_daily_carry_adjacency_matrix_2000_2019.csv', index_col=0, header=0)
graph_list = [A_t.values, A_n.values]

# Training vs Evaluation
split = '2015-01-01'
rs_y = rs_y[split:]

# Evaluate over episodes

'''
TODO:
- Iterate over models for each of the pisodes
- For each model, run it over the entire validation period
- Plot and save plot
'''

for e in range(episode_count):

    # Graph Tensor Network Agent
    agent = GTNAgent(state_size=(window_size, rs_data['carry'].shape[1], len(rs_types)),
                     graph_list=graph_list,
                     model_name=f'model_ep{e}',
                     model_target_name=f'model_target_ep{e}',
                     is_eval=True)

    # Print progress
    print(f"Episode: {e + 1}/{episode_count}")
    print(f"Epsilon: {agent.epsilon}")

    # Reset agent parameters to run next episode
    agent.episode_reset()

    # Loop over time
    for t in rs_y.index[window_size:]:

        # past {window_size} log returns up to and excluding {t}
        # X = rs_data.loc[:t].iloc[-window_size-1:-1]  # fetch raw data
        # X = X.values.reshape([1]+list(X.shape))  # tensorize
        X = np.array([rs_data[k].loc[:t].iloc[-window_size - 1:-1].values for k in rs_data.keys()])
        X = X.transpose([1, 2, 0])
        X = X.reshape([1] + list(X.shape))

        # Get action from agent
        action = agent.act(X)

        # Process returns/rewards
        action_direction = -1 * (action * 2 - 1)  # map 0->buy->+1, 1->sell->-1
        reward = 100 * action_direction * rs_y[t]
        agent.episode_tot_reward += reward
        agent.episode_rewards.append(reward)
        print(t, agent.episode_tot_reward, reward)

        # Fetch next state
        done = True if t == rs_y.index[-1] else False
        # next_X = rs_data.loc[:t].iloc[-window_size:]  # fetch raw data
        # next_X = next_X.values.reshape([1]+list(next_X.shape))  # tensorize
        next_X = np.array([rs_data[k].loc[:t].iloc[-window_size:].values for k in rs_data.keys()])
        next_X = next_X.transpose([1, 2, 0])
        next_X = next_X.reshape([1] + list(next_X.shape))

        # Append to memory & train
        # agent.memory.append((X[0], action, reward, next_X[0], done))
        # agent.replay(min(batch_size, len(agent.memory)))

        # Print if done
        if done:
            print("--------------------------------")
            print(f"Episode reward:{agent.episode_tot_reward}%")
            print("--------------------------------")

    # Record episode data
    total_rewards_df[e] = agent.episode_rewards

    # # Update agent parameters
    # agent.update_target_model()
    # agent.epsilon = max(agent.epsilon_decay * agent.epsilon, agent.epsilon_min)

    # # Save models
    # agent.model.save("models/model_ep" + str(e))
    # agent.target_model.save("models/model_target_ep" + str(e))

plt.figure()
total_rewards_df.cumsum().plot()
plt.savefig('results/(EVALUATION) strategy returns.png')

plt.figure()
total_rewards_df.sum().plot()
plt.savefig('results/(EVALUATION) total returns across episodes.png')

