import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt


plt.style.use('fivethirtyeight')
files = glob.glob( 'results/**.csv' )


# Returns averaged across models per episodes plotting
ep = '14'
col_names = [(file.split('GTN ')[-1]).split(' testing')[0] for file in files if 'testing' in file]

testing_df = [pd.read_csv(file, index_col=0, header=0)[ep] for file in files if 'testing' in file]
testing_df = pd.concat(testing_df, axis=1)
testing_df.columns = col_names

sel = ['EURUSD', 'USDCHF', 'GBPUSD', 'USDNOK', 'USDSEK'] # european currencies only
testing_df[sel].mean(1).cumsum().plot()