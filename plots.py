import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt

ep = '15'
files = glob.glob( 'results/**.csv' )
col_names = [(file.split('GTN ')[-1]).split('testing')[0] for file in files if 'testing' in file]

testing_df = [pd.read_csv(file, index_col=0, header=0)[ep] for file in files if 'testing' in file]
testing_df = pd.concat(testing_df, axis=1)
testing_df.columns = col_names

training_df = [pd.read_csv(file, index_col=0, header=0)[ep] for file in files if 'training' in file]
training_df = pd.concat(training_df, axis=1)
training_df.columns = col_names

testing_df.mean(1).cumsum().plot()