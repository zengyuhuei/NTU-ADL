import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

with open('json/dqn_learning_curve.json') as f:
    learning_curve = json.load(f)

with open('json/ddqn_learning_curve.json') as f:
    learning_curve_1000 = json.load(f)



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


plt.figure(figsize=[15,10])
plt.grid(True)
plt.plot(df['pandas_SMA_150'],label='DQN')
plt.plot(df_1000['pandas_SMA_150'],label='DDQN')
plt.legend(loc=2)
plt.title('DQN vs DDQN (window = 150)')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.savefig("plot/dqn_ddqn_cp.png")
plt.show()

