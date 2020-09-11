# Path of datasets 
TRAIN_PATH = './data/train.json'
DEV_PATH = './data/dev.json'
TEST_PATH = './data/test.json'
DATA_FOLDER = './data/'

# 指定繁簡中文 BERT-BASE 預訓練模型
PRETRAINED_MODEL_NAME = "bert-base-chinese"  

# Special Tokens
CLS = '[CLS]'
SEP = '[SEP]'
UNK = '[UNK]'
PAD = '[PAD]'
MASK = '[MASK]'

DEVICE = 'cuda:0'
BATCH_SIZE = 8
MAX_LENGTH = 512
EPOCHS = 10