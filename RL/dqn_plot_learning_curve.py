import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
with open('dqn_learning_curve.json') as f:
    learning_curve = json.load(f)



data = {}
data['episode'] = []
data['reward'] = []
for key, value in learning_curve.items():
    #if (int(key) % 10) == 0:
    data['episode'].append(key)
    data['reward'].append(value)
df = pd.DataFrame(data)

'''
for i in range(0,df.shape[0]-19):
    df.loc[df.index[i+19],'SMA_20'] = np.round(((df.iloc[i,1]+ df.iloc[i+1,1] +df.iloc[i+2,1]
                                                +df.iloc[i+3,1]+ df.iloc[i+4,1] +df.iloc[i+5,1]
                                                +df.iloc[i+6,1]+ df.iloc[i+7,1] +df.iloc[i+8,1]
                                                +df.iloc[i+9,1]+ df.iloc[i+10,1] +df.iloc[i+11,1]
                                                +df.iloc[i+12,1]+ df.iloc[i+13,1] +df.iloc[i+14,1]
                                                +df.iloc[i+15,1]+ df.iloc[i+16,1] +df.iloc[i+17,1]
                                                +df.iloc[i+18,1]+ df.iloc[i+19,1])/20),1)
df['pandas_SMA_20'] = df.iloc[:,1].rolling(window=20).mean()
'''

for i in range(0,df.shape[0]-149):
    total = 0
    for j in range(0,150):
        total += df.iloc[i+j,1]
    df.loc[df.index[i+149],'SMA_150'] = np.round((total/150),1)
df['pandas_SMA_150'] = df.iloc[:,1].rolling(window=150).mean()

plt.figure(figsize=[15,10])
plt.grid(True)
plt.plot(df['reward'],label='average reward',alpha=0.5)
plt.plot(df['pandas_SMA_150'],label='SMA 150 average reward')
plt.legend(loc=2)
plt.title('Learning Curve of Reward (window = 150)')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.savefig("dqn_learning_curve.png")
plt.show()

