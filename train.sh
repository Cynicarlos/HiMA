#!/bin/bash

# 指定要使用的 Conda 环境
ENV_NAME="hima"

# 指定要使用的 GPU 编号
GPU_ID=0

# 训练脚本的名称
TRAIN_SCRIPT="train.py --config=configs/sony.yaml"

# 创建一个新的 tmux 会话
SESSION_NAME="hima"
tmux new-session -d -s $SESSION_NAME

# 激活 Conda 环境
tmux send-keys -t $SESSION_NAME "source activate ${ENV_NAME}" C-m

# 设置环境变量，指定要使用的 GPU
tmux send-keys -t $SESSION_NAME "export CUDA_VISIBLE_DEVICES=${GPU_ID}" C-m

# 运行训练脚本
tmux send-keys -t $SESSION_NAME "python ${TRAIN_SCRIPT} > train.log 2>&1" C-m

# 如果你稍后想要重新连接到 tmux 会话，可以使用以下命令：
# tmux attach-session -t $SESSION_NAME