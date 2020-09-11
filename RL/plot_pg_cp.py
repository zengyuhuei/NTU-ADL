import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

with open('json/pg_learning_curve.json') as f:
    learning_curve = json.load(f)

with open('json/pg_improvement_learning_curve.json') as f:
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

for i in range(0,df.shape[0]-19):
    df.loc[df.index[i+19],'SMA_20'] = np.round(((df.iloc[i,1]+ df.iloc[i+1,1] +df.iloc[i+2,1]
                                                +df.iloc[i+3,1]+ df.iloc[i+4,1] +df.iloc[i+5,1]
                                                +df.iloc[i+6,1]+ df.iloc[i+7,1] +df.iloc[i+8,1]
                                                +df.iloc[i+9,1]+ df.iloc[i+10,1] +df.iloc[i+11,1]
                                                +df.iloc[i+12,1]+ df.iloc[i+13,1] +df.iloc[i+14,1]
                                                +df.iloc[i+15,1]+ df.iloc[i+16,1] +df.iloc[i+17,1]
                                                +df.iloc[i+18,1]+ df.iloc[i+19,1])/20),1)
df['pandas_SMA_20'] = df.iloc[:,1].rolling(window=20).mean()

for i in range(0,df_1000.shape[0]-19):
    df_1000.loc[df_1000.index[i+19],'SMA_20'] = np.round(((df_1000.iloc[i,1]+ df_1000.iloc[i+1,1] +df_1000.iloc[i+2,1]
                                                +df_1000.iloc[i+3,1]+ df_1000.iloc[i+4,1] +df_1000.iloc[i+5,1]
                                                +df_1000.iloc[i+6,1]+ df_1000.iloc[i+7,1] +df_1000.iloc[i+8,1]
                                                +df_1000.iloc[i+9,1]+ df_1000.iloc[i+10,1] +df_1000.iloc[i+11,1]
                                                +df_1000.iloc[i+12,1]+ df_1000.iloc[i+13,1] +df_1000.iloc[i+14,1]
                                                +df_1000.iloc[i+15,1]+ df_1000.iloc[i+16,1] +df_1000.iloc[i+17,1]
                                                +df_1000.iloc[i+18,1]+ df_1000.iloc[i+19,1])/20),1)
df_1000['pandas_SMA_20'] = df_1000.iloc[:,1].rolling(window=20).mean()


plt.figure(figsize=[15,10])
plt.grid(True)
plt.plot(df['pandas_SMA_20'],label='Policy Gradient')
plt.plot(df_1000['pandas_SMA_20'],label='Improvement of PG')
plt.legend(loc=2)
plt.title('Policy Gradient (window = 20)')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.savefig("plot/pg_cp.png")
plt.show()

