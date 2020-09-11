#!/bin/bash

#alias python='python3.7'

TEST_INPUT_PATH="${1}"
PREDICT_OUPUT_PATH="${2}"
TEST_OUPUT_PATH="datasets/seq_tag/test.pkl"
EMBEDDING_FILE_PATH="datasets/seq_tag/embedding.pkl"
MODEL_PATH="src/model_state/seq_tag/ckpt.10.pt"
python3.7 src/preprocess_seq_tag_test.py $TEST_INPUT_PATH \
                                  $TEST_OUPUT_PATH \
                                  $EMBEDDING_FILE_PATH
python3.7 src/eval_seq_tag.py $TEST_OUPUT_PATH $PREDICT_OUPUT_PATH $EMBEDDING_FILE_PATH $MODEL_PATH

