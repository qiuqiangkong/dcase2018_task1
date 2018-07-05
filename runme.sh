#!/bin/bash
# You need to modify this path
DATASET_DIR="/vol/vssp/datasets/audio/dcase2018/task1"

# You need to modify this path as your workspace
WORKSPACE="/vol/vssp/msos/qk/workspaces/dcase2018_task1"

DEV_SUBTASK_A_DIR="TUT-urban-acoustic-scenes-2018-development"
DEV_SUBTASK_B_DIR="TUT-urban-acoustic-scenes-2018-mobile-development"
LB_SUBTASK_A_DIR="TUT-urban-acoustic-scenes-2018-leaderboard"
LB_SUBTASK_B_DIR="TUT-urban-acoustic-scenes-2018-mobile-leaderboard"

# Extract features
python features.py logmel --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_A_DIR --data_type=development --workspace=$WORKSPACE
python features.py logmel --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_B_DIR --data_type=development --workspace=$WORKSPACE
python features.py logmel --dataset_dir=$DATASET_DIR --subdir=$LB_SUBTASK_A_DIR --data_type=leaderboard --workspace=$WORKSPACE
python features.py logmel --dataset_dir=$DATASET_DIR --subdir=$LB_SUBTASK_B_DIR --data_type=leaderboard --workspace=$WORKSPACE

########################
# Train model for subtask A
CUDA_VISIBLE_DEVICES=0 python main.py train --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_A_DIR --workspace=$WORKSPACE --validation=True

# Evaluate subtask A
CUDA_VISIBLE_DEVICES=0 python main.py inference_validation --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_A_DIR --workspace=$WORKSPACE --iteration=1000

########################
# Train model for subtask B
CUDA_VISIBLE_DEVICES=0 python main.py train --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_B_DIR --workspace=$WORKSPACE --validation=True

# Evaluate subtask B
CUDA_VISIBLE_DEVICES=0 python main.py inference_validation --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_B_DIR --workspace=$WORKSPACE --iteration=3000