#!/bin/bash

MODEL_URL="https://www.dropbox.com/s/8pwq4ps02vuim6y/ckpt.2.pkl?dl=1"
MODEL_PATH="model_state/ckpt.2.pkl"
wget $MODEL_URL -O $MODEL_PATH
