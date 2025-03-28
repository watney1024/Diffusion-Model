#!/bin/bash

# 确保按顺序运行
nohup python3 train.py > train.log 2>&1 &
wait $!  # 等待 train.py 执行完毕

nohup python3 train_1.py > train1.log 2>&1 &
wait $!  # 等待 train1.py 执行完毕

nohup python3 train_2.py > train2.log 2>&1 &
wait $!  # 等待 train2.py 执行完毕