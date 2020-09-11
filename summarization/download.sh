#!/bin/bash

EMBEDDING_URL="https://www.dropbox.com/s/ul3upb9c7j04q5c/embedding.pkl?dl=1"
EXTRACTIVE_MODEL_URL="https://www.dropbox.com/s/xsdxm53l6ampsb1/ckpt.10.pt?dl=1"
EMBEDDING_FILE_PATH="datasets/seq_tag/embedding.pkl"
EXTRACTIVE_MODEL_PATH="src/model_state/seq_tag/ckpt.10.pt"

SEQ2SEQ_EMBEDDING_URL="https://www.dropbox.com/s/fi8ea1gjazpo2b0/embedding.pkl?dl=1"
SEQ2SEQ_MODEL_URL="https://www.dropbox.com/s/6v06qm5oq67k9he/ckpt.6.pt?dl=1"
SEQ2SEQ_FILE_PATH="datasets/seq2seq/embedding.pkl"
SEQ2SEQ_MODEL_PATH="src/model_state/seq2seq/ckpt.6.pt"

ATTENTION_EMBEDDING_URL="https://www.dropbox.com/s/8rhopqj565h3r13/embedding.pkl?dl=1"
ATTENTION_MODEL_URL="https://www.dropbox.com/s/a4i0rfccwxk7y8m/ckpt.20.pt?dl=1"
ATTENTION_FILE_PATH="datasets/attention/embedding.pkl"
ATTENTION_MODEL_PATH="src/model_state/attention/ckpt.20.pt"

wget $EMBEDDING_URL -O $EMBEDDING_FILE_PATH
wget $EXTRACTIVE_MODEL_URL -O $EXTRACTIVE_MODEL_PATH

wget $SEQ2SEQ_EMBEDDING_URL -O $SEQ2SEQ_FILE_PATH
wget $SEQ2SEQ_MODEL_URL -O $SEQ2SEQ_MODEL_PATH

wget $ATTENTION_EMBEDDING_URL -O $ATTENTION_FILE_PATH
wget $ATTENTION_MODEL_URL -O $ATTENTION_MODEL_PATH