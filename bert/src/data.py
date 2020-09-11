
import json
from tqdm import tqdm
from transformers import BertTokenizer

import torch
from torch.utils.data import Dataset
import data_util as data_util
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

DEVICE = 'cuda:0'
MAX_LENGTH = 512
class QADataset(Dataset):
    def __init__(self, mode, path, tokenizer):
        self.mode = mode
        self.raw_data = data_util.raw_data(self.mode, path)
        self.tokenizer = tokenizer
        if mode == 'train':
            print('origin data_num: {}'.format(len(self.raw_data)))
            self.raw_data = self.filter_data(self.raw_data)
            print('Afert filter data_num: {}'.format(len(self.raw_data)))
        self.len = len(self.raw_data)


    def filter_data(self, raw_datas):
        filted = []
        progress = tqdm(total=len(raw_datas))
        for raw_data in raw_datas:
            progress.update(1)
            context_token = self.tokenizer.tokenize(raw_data['context'])
            question_token = self.tokenizer.tokenize(raw_data['question'])
            is_index_match = data_util.check_index(raw_data, context_token)
            if is_index_match == False:
                continue
            ns, ne = data_util.index_after_tokenize(context_token, raw_data['answer_start'], raw_data['answer_end'])
            r = self.tokenizer.prepare_for_model(
                context_token, question_token, max_length=MAX_LENGTH,
                truncation_strategy='only_first', pad_to_max_length=True)
            input_ids = r['input_ids']
            token_type_ids = r['token_type_ids']
            attention_mask = r['attention_mask']
            context_end = 0
            for i, pos in enumerate(token_type_ids):
                if pos == 1:
                    context_end = i
                    break
            if (ns > context_end or ne > context_end) and (ns < 512 and ne < 512):
                continue
            filted.append(raw_data)
        progress.close()
        return filted

    def __getitem__(self, idx):
        id_ = self.raw_data[idx]['id']
        if self.mode == "test":
            context, question = self.raw_data[idx]['context'], self.raw_data[idx]['question']
            return (context, question, id_)
        else:
            context, question, answerable, answer_start, answer_end = self.raw_data[idx]['context'], self.raw_data[
                idx]['question'], self.raw_data[idx]['answerable'], self.raw_data[
                    idx]['answer_start'], self.raw_data[idx]['answer_end']

            return (context, question, answerable, answer_start, answer_end, id_)

    def __len__(self):
        return self.len

    def collate_fn(self, samples):
        
        context = [s[0] for s in samples]
        context_token = [self.tokenizer.tokenize(s[0]) for s in samples]
        context_token_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in context_token]
        question = [s[1] for s in samples]
        question_token = [self.tokenizer.tokenize(s[1]) for s in samples]
        question_token_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in question_token]

        _ids = [s[-1] for s in samples]
        
            
        
        if self.mode != 'test':
            answerable = [s[2] for s in samples]
            answer_start = [s[3] for s in samples]
            answer_end = [s[4] for s in samples]
            new_start = []
            new_end = []
            for i in range(len(samples)):
                if answerable[i] == True:
                    #print(answer_start[i], answer_end[i])
                    ns, ne = data_util.index_after_tokenize(
                        context_token[i], 
                        answer_start[i], 
                        answer_end[i])
                    new_start.append(ns)
                    new_end.append(ne)
                    # add [CLS] length
                    new_start[i] += 1
                    new_end[i] += 1
                else:
                    new_start.append(-1)
                    new_end.append(-1)
            answerable = torch.tensor(answerable, dtype=torch.float, device=DEVICE)
            new_start = torch.tensor(new_start, device=DEVICE)
            new_end = torch.tensor(new_end, device=DEVICE)


        # truncate and padding
        input_ids = []
        token_type_ids = []
        attention_mask = []
        for i in range(len(samples)):
            r = self.tokenizer.prepare_for_model(
                context_token_ids[i], question_token_ids[i], max_length=MAX_LENGTH,
                truncation_strategy='only_first', pad_to_max_length=True)
            input_ids.append(r['input_ids'])
            token_type_ids.append(r['token_type_ids'])
            attention_mask.append(r['attention_mask'])


        # from list to tensor
        for i in range(len(samples)):
            input_ids[i] = torch.tensor(input_ids[i], device=DEVICE)
            token_type_ids[i] = torch.tensor(token_type_ids[i], device=DEVICE)
            attention_mask[i] = torch.tensor(attention_mask[i], device=DEVICE)
        input_ids = torch.stack(input_ids)
        token_type_ids = torch.stack(token_type_ids)
        attention_mask = torch.stack(attention_mask)
        if self.mode != 'test':    
            return input_ids, token_type_ids, attention_mask, answerable, new_start, new_end, _ids
        else:
            return input_ids, token_type_ids, attention_mask, _ids


def load(mode, path):
    # 取得此預訓練模型所使用的 tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", do_lower_case=True)
    return QADataset(mode, path, tokenizer=tokenizer)
