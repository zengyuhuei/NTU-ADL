from const import TRAIN_PATH, DEV_PATH, TEST_PATH, DATA_FOLDER, DEVICE, MAX_LENGTH
import json
from tqdm import tqdm
from transformers import BertTokenizer
from const import PRETRAINED_MODEL_NAME
import torch


def load_json(path):
    with open(path) as f:
        data = json.loads(f.read())
    return data


def raw_data(mode,path):
    data = load_json(path)['data']
    dataset = []
    for i in tqdm(range(len(data))):
        paragraphs = data[i]['paragraphs']
        for paragraph in paragraphs:
            context = paragraph['context'].replace(' ', '_').replace('　', '_')
            qas = paragraph['qas']
            for qa in qas:
                _id = qa['id']
                question = qa['question']
                if mode != 'test':
                    answerable = int(qa['answerable'])
                    answer_start = 512
                    answer_end = 512
                    #if len(context) + len(question) > MAX_LENGTH - 10:
                    #    continue
                    if qa['answerable']:
                        answer = qa['answers'][0]
                        answer_text = answer['text']
                        answer_start = answer['answer_start']
                        answer_end = answer['answer_start'] + \
                            len(answer['text']) - 1
                    dataset.append({
                        'id': _id,
                        'context': context,
                        'question': question,
                        'answerable': answerable,
                        'answer_text': answer_text,
                        'answer_start': answer_start,
                        'answer_end': answer_end
                    })
                else:
                    dataset.append({
                        'id': _id,
                        'context': context,
                        'question': question,
                    })
    return dataset


def index_after_tokenize(tokens, start, end):
    char_count, new_start, new_end = 0, 512, 512
    for i, token in enumerate(tokens):
        if token == '[UNK]' or token == '[CLS]' or token == '[SEP]':
            if char_count == start:
                new_start = i
            if char_count == end:
                new_end = i
            char_count += 1
        else:
            for c in token:
                if char_count == start:
                    new_start = i
                if char_count == end:
                    new_end = i
                if c != '#':
                    char_count += 1
    return new_start, new_end

def check_index(raw_data, tokens):
    answerable = raw_data['answerable']
    s, e = raw_data['answer_start'], raw_data['answer_end']
    text = raw_data['answer_text']
    ns, ne = index_after_tokenize(tokens, s, e)
    n_text = combine_tokens(tokens, ns, ne)
    #n_text = n_text.replace('[UNK]','#')
    #for i, c in enumerate(n_text):
    #    if c == '#':
    #        n_text = n_text.replace('#', text[i])
    #print(tokens)
    #print(text, n_text)
    if text == n_text or n_text in text or answerable == False:
        return True
    return False

def combine_tokens(tokens, ns, ne):
    n_text = ''
    for i in range(ns, min(ne+1, len(tokens))):
        n_text += tokens[i]
    n_text = n_text.replace('#','').replace('[UNK]','?').replace('[SEP]','')
    return n_text
