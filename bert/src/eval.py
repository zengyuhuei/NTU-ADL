import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import data as data
import data_util as data_util
from model import BertForQuestionAnsweringCustom
from tqdm import tqdm
from torch import nn
import json
import argparse


def eval(model, dataloader):
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", do_lower_case=True)
    pred = {}
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress = tqdm(total=len(dataloader))
        for data in dataloader:
            progress.update(1)
            if next(model.parameters()).is_cuda:
                dataset = [t.to(DEVICE) for t in data[:-1] if t is not None]

            _id = data[-1]
            
            input_ids, token_type_ids, attention_mask = dataset[:]
            
            outputs = model(input_ids=input_ids, 
                            attention_mask=attention_mask, 
                            token_type_ids=token_type_ids)

            softmax = nn.Softmax(dim = 1)
            context_mask = (1-token_type_ids) * attention_mask
            for i in range(len(context_mask)):
                for j in range(len(context_mask[i])):
                    if context_mask[i][j].item() == 0:
                        outputs[1][i][j] = float("-inf")
                        outputs[2][i][j] = float("-inf")
            front = softmax(outputs[1])
            back = softmax(outputs[2])
            
            
            _cls = outputs[0]
            answerable_threshold = 0.5
            predict_answerable = [1 if d > answerable_threshold else 0 for d in _cls]
            for i in range(len(context_mask)):
                outputs[1][i][0] = float("-inf")
                outputs[2][i][0] = float("-inf")

            _, i_start = front.max(1)
            _, i_back = back.max(1)

            context_end = []
            for b in token_type_ids:
                for i, pos in enumerate(b):
                    if pos == 1:
                        context_end.append(i)
                        break

            for i in range(len(predict_answerable)):
                if predict_answerable[i]:
                    context = input_ids[i][:context_end[i]]
                    context_token = tokenizer.convert_ids_to_tokens(context)
                    
                    ans = data_util.combine_tokens(context_token, i_start[i], i_back[i])
                    
                    answer = ans
                    if i_start[i] == 512 or i_back[i] == 512 or ((i_back[i] - i_start[i]) > 60):
                        answer = '' 
                    
                else:
                    answer =""
                pred[_id[i]] = answer
    return pred

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('test_data_path')
    parser.add_argument('output_path')
    parser.add_argument('model_path')
    args = parser.parse_args()
    return args   

if __name__ == '__main__':
    args = _parse_args()
    BATCH_SIZE = 8
    DEVICE = 'cuda:0'
    TEST_PATH = args.test_data_path
    test = data.load('test', TEST_PATH)

    testloader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=test.collate_fn)

    model = BertForQuestionAnsweringCustom.from_pretrained("bert-base-chinese")
    model = model.to(DEVICE) 
    model.load_state_dict(torch.load(args.model_path))
        
    pred = eval(model, testloader)
    output_file = open(args.output_path, 'w')       


    output_file.write(json.dumps(pred))
    output_file.close()
