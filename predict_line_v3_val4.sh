#!/usr/bin/env python
# -*- codinbg: utf-8 -*-

# v3_val4_init_120pth model in train set: 3_1
python predict.py -c --mask-threshold 0.5 --model log_line_v3_val4_init/checkpoints/CP120.pth --input data/dataset_line_v2_train_3_1/png --output pred_test_line_v2_10b/v3_val4_init_120pth_3_1 --no-crf --task-type line
# v3_val4_init_120pth model in test set: 4_2
python predict.py -c --mask-threshold 0.5 --model log_line_v3_val4_init/checkpoints/CP120.pth --input data/dataset_line_v2_train_4_2/png --output pred_test_line_v2_10b/v3_val4_init_120pth_4_2 --no-crf --task-type line
# v3_val4_init_120pth model in train set: 5_1
python predict.py -c --mask-threshold 0.5 --model log_line_v3_val4_init/checkpoints/CP120.pth --input data/dataset_line_v2_train_5_1/png --output pred_test_line_v2_10b/v3_val4_init_120pth_5_1 --no-crf --task-type line
# v3_val4_init_120pth model in train set: 10_1
python predict.py -c --mask-threshold 0.5 --model log_line_v3_val4_init/checkpoints/CP120.pth --input data/dataset_line_v2_test_10_1/png --output pred_test_line_v2_10b/v3_val4_init_120pth_10_1 --no-crf --task-type line
