#!/bin/bash

#alias python='python3.7'

TEST_INPUT_PATH="${1}"
PREDICT_OUPUT_PATH="${2}"
MODEL_PATH="model_state/ckpt.2.pkl"
python3.7 src/eval.py $TEST_INPUT_PATH $PREDICT_OUPUT_PATH $MODEL_PATH




