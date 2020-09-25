import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt


def get_testing_df (agent='GTN', ep='14'):

    # Returns averaged across models per episodes plotting
    files = glob.glob('results/**.csv')
    files = [file for file in files if ('testing' in file) and (agent in file)]
    col_names = [(n.split(f'{agent} ')[-1]).split(' testing')[0] for n in files]

    testing_df = [pd.read_csv(file, index_col=0, header=0)[ep] for file in files if 'testing' in file]
    testing_df = pd.concat(testing_df, axis=1)
    testing_df.columns = col_names

    # european currencies only
    sel = ['EURUSD', 'USDCHF', 'GBPUSD', 'USDNOK', 'USDSEK']
    testing_df = testing_df[sel]

    return testing_df


def compute_metrics (rs):

    # Total return
    total_r = rs.sum()

    # Sharpe ratio
    sharpe = rs.mean()/rs.std()

    # Maximum Draw-Down
    current_sum = 0
    max_sum = 0
    for n in -rs:
        current_sum = max(0, current_sum + n)
        max_sum = max(current_sum, max_sum)
    max_dd = max_sum

    # Hit ratio
    hit_ratio = rs.apply(np.sign).replace(-1,0).mean()

    # Results
    results = pd.Series(data=[total_r, sharpe, max_dd, hit_ratio],
                        index=['Total Return', 'Sharpe', 'Max DD', 'Hit Ratio'])

    return results


agents = ['GTN', 'RNN', 'TTNN', 'GNN']
all_rs = []
for agent in agents:
    all_rs.append(get_testing_df(agent=agent).mean(1).copy())

plt.rcParams['font.size'] = 20
all_cumrs = 1000*((0.01*pd.concat(all_rs, axis=1)).cumsum().apply(np.exp))
all_cumrs.plot(figsize=(12, 6), linewidth=5, grid=True)
plt.legend(agents)
plt.title('Out-of-Sample Performance')
plt.xlabel('minutes')
plt.ylabel('portfolio value')
plt.tight_layout()

results = pd.concat([compute_metrics(rs) for rs in all_rs], axis=1)
results.columns = agents
print(results)