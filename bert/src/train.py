import torch
from transformers import BertForQuestionAnswering, BertTokenizer
from model import BertForQuestionAnsweringCustom
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch import nn
from torch import autograd
from tqdm import tqdm
import data as data

DEVICE = 'cuda:0'
BATCH_SIZE = 8
MAX_LENGTH = 512
EPOCHS = 5

def eval(model, dataloader):
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", do_lower_case=True)
    predictions = None
    correct = 0
    anserable_correct = 0
    total = 0
    
    with torch.no_grad():
        print("predictings")
        progress = tqdm(total=len(dataloader))
        for data in dataloader:
            progress.update(1)
            tokens_tensors, segments_tensors, masks_tensors, answerable, answer_start, answer_end  = [t.to(DEVICE) for t in data[:-1]]
            outputs = model(input_ids=tokens_tensors, 
                            attention_mask=masks_tensors, 
                            token_type_ids=segments_tensors)

            softmax = nn.Softmax()
            _cls = outputs[0]
            front = softmax(outputs[1])
            back = softmax(outputs[2])
            
            answerable_threshold = 0.5
            predict_answerable = [1 if d > answerable_threshold else 0 for d in _cls]
            labels = data[3]
            total += len(predict_answerable)
            anserable_correct += sum([1 if d[0] == d[1] else 0 for d in zip(predict_answerable, labels)])
           
        progress.close()
    acc = anserable_correct / total
    return acc


def train():
    train_path = '../data/train.json'
    train = data.load('train', train_path)
    trainloader = DataLoader(train, batch_size=BATCH_SIZE, shuffle = True,
                            collate_fn=train.collate_fn)
    model = BertForQuestionAnsweringCustom.from_pretrained("bert-base-chinese")
    model = model.to(DEVICE)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    for epoch in range(EPOCHS):
        running_loss = 0.0
        print("Training Epoch {}".format(epoch+1))
        progress = tqdm(total=len(trainloader))
        for d in trainloader:
            progress.update(1)
            input_ids, token_type_ids, attention_mask, answerable, answer_start, answer_end = [t.to(DEVICE) for t in d[:-1]]
            optimizer.zero_grad()
            # forward pass
            outputs = model(input_ids=input_ids, 
                            attention_mask=attention_mask, 
                            token_type_ids=token_type_ids, 
                            start_positions=answer_start,
                            end_positions=answer_end,
                            answerable=answerable)
            answerable_loss, start_end_loss = outputs[0], outputs[1]
         
            total_loss = answerable_loss + start_end_loss
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
            progress.set_description("Answerable loss: %.3f, StartEnd loss: %.3f" %(answerable_loss.item(), start_end_loss.item()))
            
            
        progress.close()     
        dev_path = '../data/dev.json'
        dev = data.load('dev', dev_path)
        devloader = DataLoader(dev, batch_size=BATCH_SIZE, shuffle = False,
                                collate_fn=dev.collate_fn)
        acc = eval(model, devloader)
        print('[epoch %d] loss: %.3f, acc: %.3f' %(epoch + 1, running_loss, acc))
        # save model
        print('Save model as ckpt.{}.pkl'.format(epoch + 1))
        torch.save(model.state_dict(), './model_state/ckpt.{}.pkl'.format(epoch + 1))

if __name__ == "__main__":
    train()
