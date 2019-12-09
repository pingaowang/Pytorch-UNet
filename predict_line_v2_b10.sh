#!/usr/bin/env python
# -*- codinbg: utf-8 -*-

# init model in test set: 10_1, 10_2, 10_3
#python predict.py --model log_line_v2_10b/checkpoints/CP180.pth --input data/dataset_line_v2_test_10_1/png --output pred_test_line_v2_10b_1_10_1 --no-crf --task-type line
#python predict.py --model log_line_v2_10b/checkpoints/CP180.pth --input data/dataset_line_v2_test_10_2/png --output pred_test_line_v2_10b_1_10_2 --no-crf --task-type line
#python predict.py --model log_line_v2_10b/checkpoints/CP180.pth --input data/dataset_line_v2_test_10_3/png --output pred_test_line_v2_10b_1_10_3 --no-crf --task-type line

# resume_1 model in test set: 10_1, 10_2, 10_3
#python predict.py -c --model log_line_v2_10b_resume_1/checkpoints/CP20.pth --input data/dataset_line_v2_test_10_1/png --output pred_test_line_v2_10b/resume1_1_10_1 --no-crf --task-type line
#python predict.py -c --model log_line_v2_10b_resume_1/checkpoints/CP20.pth --input data/dataset_line_v2_test_10_2/png --output pred_test_line_v2_10b/resume1_1_10_2 --no-crf --task-type line
#python predict.py -c --model log_line_v2_10b_resume_1/checkpoints/CP20.pth --input data/dataset_line_v2_test_10_3/png --output pred_test_line_v2_10b/resume1_1_10_3 --no-crf --task-type line

# init model in train set: 2_1
#python predict.py -c --model log_line_v2_10b/checkpoints/CP180.pth --input data/dataset_line_v2_train_2_1/png --output pred_test_line_v2_10b/train_init_2_1 --no-crf --task-type line

# resume_1 model in train set: 2_1
#python predict.py -c --model log_line_v2_10b_resume_1/checkpoints/CP40.pth --input data/dataset_line_v2_train_2_1/png --output pred_test_line_v2_10b/train_resume_1_2_1 --no-crf --task-type line

# init model in train set: 2_1, --mask-threshold = 0.4
#python predict.py --mask-threshold 0.4 --model log_line_v2_10b/checkpoints/CP180.pth --input data/dataset_line_v2_train_2_1/png --output pred_test_line_v2_10b/train_init_2_1_thread_04 --no-crf --task-type line

# init model in train set: 2_1, --mask-threshold = 0.2
#python predict.py --mask-threshold 0.2 --model log_line_v2_10b/checkpoints/CP180.pth --input data/dataset_line_v2_train_2_1/png --output pred_test_line_v2_10b/train_init_2_1_thread_02 --no-crf --task-type line

# init model in train set: 2_1, --mask-threshold = 0.1
#python predict.py --mask-threshold 0.1 --model log_line_v2_10b/checkpoints/CP180.pth --input data/dataset_line_v2_train_2_1/png --output pred_test_line_v2_10b/train_init_2_1_thread_01 --no-crf --task-type line

# init model in test set: 10_1, 10_2, 10_3, --mask-threshold = 0.1
#python predict.py --mask-threshold 0.1 --model log_line_v2_10b/checkpoints/CP180.pth --input data/dataset_line_v2_test_10_1/png --output pred_test_line_v2_10b/pred_test_line_v2_10b_1_10_1_thread_01 --no-crf --task-type line
#python predict.py --mask-threshold 0.1 --model log_line_v2_10b/checkpoints/CP180.pth --input data/dataset_line_v2_test_10_2/png --output pred_test_line_v2_10b/pred_test_line_v2_10b_1_10_2_thread_01 --no-crf --task-type line
#python predict.py --mask-threshold 0.1 --model log_line_v2_10b/checkpoints/CP180.pth --input data/dataset_line_v2_test_10_3/png --output pred_test_line_v2_10b/pred_test_line_v2_10b_1_10_3_thread_01 --no-crf --task-type line

# init model in test set: 10_1, 10_2, 10_3, --mask-threshold = 0.01
#python predict.py --mask-threshold 0.01 --model log_line_v2_10b/checkpoints/CP180.pth --input data/dataset_line_v2_test_10_1/png --output pred_test_line_v2_10b/pred_test_line_v2_10b_1_10_1_thread_001 --no-crf --task-type line

# init model in test set: 10_1, 10_2, 10_3, --mask-threshold = 0.001
#python predict.py --mask-threshold 0.001 --model log_line_v2_10b/checkpoints/CP180.pth --input data/dataset_line_v2_test_10_1/png --output pred_test_line_v2_10b/pred_test_line_v2_10b_1_10_1_thread_0001 --no-crf --task-type line

# init avgpool model in test set: 10_1
#python predict.py --mask-threshold 0.5 --model log_line_v2_10b_avgpool/checkpoints/CP20.pth --input data/dataset_line_v2_test_10_1/png --output pred_test_line_v2_10b/pred_test_line_v2_10b_1_10_1_avgpool --no-crf --task-type line

# resume_1 avgpool model in test set: 10_1
#python predict.py --mask-threshold 0.5 --model log_line_v2_10b_avgpool_resume_1/checkpoints/CP100.pth --input data/dataset_line_v2_test_10_1/png --output pred_test_line_v2_10b/pred_test_line_v2_10b_1_10_1_avgpool --no-crf --task-type line

# resume_1 avgpool model in train set: 2_1
#python predict.py --mask-threshold 0.5 --model log_line_v2_10b_avgpool_resume_1/checkpoints/CP100.pth --input data/dataset_line_v2_train_2_1/png --output pred_test_line_v2_10b/train_resume_1_2_1_avgpool --no-crf --task-type line

# init maxpool model in test set: 10_1 [40 epoch]
#python predict.py -c --mask-threshold 0.1 --model log_line_v2_10b_maxpool/checkpoints/CP40.pth --input data/dataset_line_v2_test_10_1/png --output pred_test_line_v2_10b/test_init_1_2_1_maxpool_40epoch --no-crf --task-type line

# init maxpool model in train set: 2_1 [40 epoch]
#python predict.py -c --mask-threshold 0.1 --model log_line_v2_10b_maxpool/checkpoints/CP40.pth --input data/dataset_line_v2_train_2_1/png --output pred_test_line_v2_10b/train_init_1_2_1_maxpool_40epoch --no-crf --task-type line

# init exp_lr_0.0001 model in test set: 10_1 [10 epoch]
python predict.py -c --mask-threshold 0.5 --model log_line_v2_10b_LRExp_0-0001/checkpoints/CP10.pth --input data/dataset_line_v2_test_10_1/png --output pred_test_line_v2_10b/test_init_explr0-0001_10_1_maxpool --no-crf --task-type line