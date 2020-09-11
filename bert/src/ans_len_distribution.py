
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import data
from matplotlib import pyplot as plt
import argparse

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    args = parser.parse_args()
    return args   

if __name__ == '__main__':
    args = _parse_args()
    BATCH_SIZE = 8
    DEVICE = 'cuda:0'
    train_path = args.data_path
    train = data.load('train', train_path)
    trainloader = DataLoader(train, batch_size=BATCH_SIZE, shuffle = True,collate_fn=train.collate_fn)
    progress = tqdm(total=len(trainloader))
    total = 0
    result = {}
    for d in trainloader:
        progress.update(1)
        input_ids, token_type_ids, attention_mask, answerable, answer_start, answer_end = [t.to(DEVICE) for t in d[:-1]]
        for i ,ans in enumerate(answerable):
            if ans:
                total += 1
                length = answer_end[i].item() - answer_start[i].item() + 1
                if length in result:
                    result[length] += 1
                else:
                    result[length] = 1
    progress.close()
    result = {i[0]:i[1] for i in sorted(result.items(), key=lambda kv: kv[0])}

    for i in result.keys():
        result[i] /= total
    max_keys = max(result.keys())
    for i in list(result):
        if i < 0:
            result.pop(i)

    for i in range(2,max_keys+1):
        if i in result:
            result[i] += result[i-1]
        else:
            result[i] = result[i-1]
    result = {i[0]:i[1] for i in sorted(result.items(), key=lambda kv: kv[0])}

    x = list(result.keys())
    y = list(result.values())

    plt.figure(figsize=(9,6))
    plt.bar(x, y, align = 'center', edgecolor = 'blue', facecolor = 'lightskyblue', alpha=0.9, width = 0.55)
    plt.title('Cumulative Answer Length')
    plt.xlabel('Length')
    plt.ylabel('Count (%)')
    plt.savefig('distribution.png')
    plt.show()
