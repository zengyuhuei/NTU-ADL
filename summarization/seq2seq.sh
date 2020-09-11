#!/bin/bash

#alias python='python3.7'

TEST_INPUT_PATH="${1}"
PREDICT_OUPUT_PATH="${2}"
TEST_OUPUT_PATH="datasets/seq2seq/test.pkl"
EMBEDDING_FILE_PATH="datasets/seq2seq/embedding.pkl"
MODEL_PATH="src/model_state/seq2seq/ckpt.6.pt"
python3.7 src/preprocess_seq2seq_test.py $TEST_INPUT_PATH \
                                  $TEST_OUPUT_PATH \
                                  $EMBEDDING_FILE_PATH
python3.7 src/eval_seq2seq.py $TEST_OUPUT_PATH $PREDICT_OUPUT_PATH $EMBEDDING_FILE_PATH $MODEL_PATH




