import json

import matplotlib.pyplot as plt
def load_json(path):
    with open(path) as f:
        data = json.loads(f.read())
    return data



answerable_threshold = [0.1, 0.3, 0.5, 0.7, 0.9]

data = []
for threshold in answerable_threshold:
    path = './threshold/result/'+str(threshold)+'_result.json'
    data.append(load_json(path))
f1_overall = []
f1_ans = []
f1_unans = []

em_overall = []
em_ans = []
em_unans = []

for i in data:
    em_overall.append(i['overall']['em'])
    f1_overall.append(i['overall']['f1'])
    em_ans.append(i['answerable']['em'])
    f1_ans.append(i['answerable']['f1'])
    em_unans.append(i['unanswerable']['em'])
    f1_unans.append(i['unanswerable']['f1'])




fig, (ax1, ax2) = plt.subplots(1, 2,sharex=True, sharey=True)
fig.suptitle('Performance on Different Threshold')
ax1.set_title('F1')
ax1.set_xticks(answerable_threshold)
ax1.plot(answerable_threshold,f1_overall,'o-',color = 'blue', label="overall")
ax1.plot(answerable_threshold,f1_ans,'o-',color = 'orange', label="answerable")
ax1.plot(answerable_threshold,f1_unans,'o-',color = 'green', label="unanswerable")

ax2.set_title('EM')
ax2.set_xticks(answerable_threshold)
ax2.plot(answerable_threshold,em_overall,'o-',color = 'blue', label="overall")
ax2.plot(answerable_threshold,em_ans,'o-',color = 'orange', label="answerable")
ax2.plot(answerable_threshold,em_unans,'o-',color = 'green', label="unanswerable")
fig.text(0.5, 0.04, 'Anserable Threshold', va='center', ha='center')

axLine, axLabel = ax2.get_legend_handles_labels()
fig.legend(axLine, axLabel, loc = 'upper right')
plt.savefig('threshold.png')
plt.show() 
#fig.set_xlabel('Anserable Threshold')
