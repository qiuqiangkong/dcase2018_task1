#!/bin/bash
# You need to modify this path
DATASET_DIR="/vol/vssp/datasets/audio/dcase2018/task1"

# You need to modify this path as your workspace
WORKSPACE="/vol/vssp/msos/qk/workspaces/pub_dcase2018_task1"

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
CUDA_VISIBLE_DEVICES=0 python main_pytorch.py train --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_A_DIR --workspace=$WORKSPACE --validate --cuda

# Evaluate subtask A
CUDA_VISIBLE_DEVICES=0 python main_pytorch.py inference_validation --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_A_DIR --workspace=$WORKSPACE --iteration=1000 --cuda

########################
# Train model for subtask B
CUDA_VISIBLE_DEVICES=0 python main_pytorch.py train --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_B_DIR --workspace=$WORKSPACE --validate --cuda

# Evaluate subtask B
CUDA_VISIBLE_DEVICES=0 python main_pytorch.py inference_validation --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_B_DIR --workspace=$WORKSPACE --iteration=3000 --cuda

########################
CUDA_VISIBLE_DEVICES=2 python main_pytorch.py train --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_A_DIR --workspace=$WORKSPACE --cuda

CUDA_VISIBLE_DEVICES=2 python main_pytorch.py inference_testing_data --dataset_dir=$DATASET_DIR --dev_subdir=$DEV_SUBTASK_A_DIR --test_subdir=$LB_SUBTASK_A_DIR --workspace=$WORKSPACE --iteration --cuda