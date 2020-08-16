#!/bin/bash

mkdir mnistCUDNN_logs

bash bi_make.sh

./train_bi 2>&1 | tee ./mnistCUDNN_logs/test_mnistCUDNN_training.log