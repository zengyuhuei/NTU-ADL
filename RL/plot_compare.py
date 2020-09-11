import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

with open('json/dqn_learning_curve_500.json') as f:
    learning_curve = json.load(f)

with open('json/dqn_learning_curve.json') as f:
    learning_curve_1000 = json.load(f)

with open('json/dqn_learning_curve_1500.json') as f:
    learning_curve_1500 = json.load(f)

with open('json/dqn_learning_curve_2000.json') as f:
    learning_curve_2000 = json.load(f)



data = {}
data['episode'] = []
data['reward'] = []

for key, value in learning_curve.items():
    #if (int(key) % 10) == 0:
    data['episode'].append(key)
    data['reward'].append(value)
df = pd.DataFrame(data)

data_1000 = {}
data_1000['episode'] = []
data_1000['reward'] = []

for key, value in learning_curve_1000.items():
    #if (int(key) % 10) == 0:
    data_1000['episode'].append(key)
    data_1000['reward'].append(value)
df_1000 = pd.DataFrame(data_1000)

data_1500 = {}
data_1500['episode'] = []
data_1500['reward'] = []

for key, value in learning_curve_1500.items():
    #if (int(key) % 10) == 0:
    data_1500['episode'].append(key)
    data_1500['reward'].append(value)
df_1500 = pd.DataFrame(data_1500)

data_2000 = {}
data_2000['episode'] = []
data_2000['reward'] = []

for key, value in learning_curve_2000.items():
    #if (int(key) % 10) == 0:
    data_2000['episode'].append(key)
    data_2000['reward'].append(value)
df_2000 = pd.DataFrame(data_2000)

for i in range(0,df.shape[0]-149):
    total = 0
    for j in range(0,150):
        total += df.iloc[i+j,1]
    df.loc[df.index[i+149],'SMA_150'] = np.round((total/150),1)
df['pandas_SMA_150'] = df.iloc[:,1].rolling(window=150).mean()

for i in range(0,df_1000.shape[0]-149):
    total = 0
    for j in range(0,150):
        total += df_1000.iloc[i+j,1]
    df_1000.loc[df_1000.index[i+149],'SMA_150'] = np.round((total/150),1)
df_1000['pandas_SMA_150'] = df_1000.iloc[:,1].rolling(window=150).mean()

for i in range(0,df_1500.shape[0]-149):
    total = 0
    for j in range(0,150):
        total += df_1500.iloc[i+j,1]
    df_1500.loc[df_1500.index[i+149],'SMA_150'] = np.round((total/150),1)
df_1500['pandas_SMA_150'] = df_1500.iloc[:,1].rolling(window=150).mean()

for i in range(0,df_2000.shape[0]-149):
    total = 0
    for j in range(0,150):
        total += df_2000.iloc[i+j,1]
    df_2000.loc[df_2000.index[i+149],'SMA_150'] = np.round((total/150),1)
df_2000['pandas_SMA_150'] = df_2000.iloc[:,1].rolling(window=150).mean()

plt.figure(figsize=[15,10])
plt.grid(True)
plt.plot(df['pandas_SMA_150'],label='frquency = 500')
#plt.plot(df['reward'],label='reward 500')
plt.plot(df_1000['pandas_SMA_150'],label='frquency = 1000')
#plt.plot(df_1000['reward'],label='reward 1000')
plt.plot(df_1500['pandas_SMA_150'],label='frquency = 1500')
#plt.plot(df_1500['reward'],label='reward 1500')
plt.plot(df_2000['pandas_SMA_150'],label='frquency = 2000')
#plt.plot(df_2000['reward'],label='reward 2000')
plt.legend(loc=2)
plt.title('Target Network Update Frequency of DQN (window = 150)')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.savefig("plot/dqn_update_freq_cp.png")
plt.show()

