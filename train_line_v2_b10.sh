#!/usr/bin/env python
# -*- codinbg: utf-8 -*-

# train with data_line_v2, BCELoss, SGD
#python train.py -l 0.1 -d 0.98 -e 1000 -b 2 -g --task-type line --data data/dataset_line_v2 --log log_line_v2_10b
#python train.py -l 0.001 -d 0.95 -e 100 -b 2 -g --task-type line --data data/dataset_line_v2 --log log_line_v2_10b_resume_1 -c log_line_v2_10b/checkpoints/CP180.pth

# use AvgPool (changed in code: unet_parts.py: line 41), MSELoss, AdamOpt
#python train.py -l 0.1 -d 0.97 -e 100 -b 2 -g --task-type line --data data/dataset_line_v2 --log log_line_v2_10b_avgpool
#python train.py -l 0.01 -d 0.95 -e 100 -b 2 -g --task-type line --data data/dataset_line_v2 --log log_line_v2_10b_avgpool_resume_1 -c log_line_v2_10b_avgpool/checkpoints/CP20.pth

# use MaxPool (changed in code: unet_parts.py: line 41), MSELoss, AdamOpt
#python train.py -l 0.01 -d 0.95 -e 50 -b 2 -g --task-type line --data data/dataset_line_v2 --log log_line_v2_10b_maxpool
#python train.py -l 0.001 -d 0.97 -e 100 -b 2 -g --task-type line --data data/dataset_line_v2 --log log_line_v2_10b_maxpool_resume_1 -c log_line_v2_10b_maxpool/checkpoints/CP50.pth

## lr exp: MaxPool, BCELoss, Adam
# lr = 0.1
#python train.py -l 0.1 -d 1 -e 40 -b 2 -g --task-type line --data data/dataset_line_v2 --log log_line_v2_10b_LRExp_0-1
# lr = 0.01
#python train.py -l 0.01 -d 1 -e 40 -b 2 -g --task-type line --data data/dataset_line_v2 --log log_line_v2_10b_LRExp_0-01
# lr = 0.001
#python train.py -l 0.001 -d 1 -e 40 -b 2 -g --task-type line --data data/dataset_line_v2 --log log_line_v2_10b_LRExp_0-001
# lr = 0.0001
#python train.py -l 0.0001 -d 1 -e 40 -b 2 -g --task-type line --data data/dataset_line_v2 --log log_line_v2_10b_LRExp_0-0001
python train.py -l 0.00001 -d 1 -e 40 -b 2 -g --task-type line --data data/dataset_line_v2 --log log_line_v2_10b_LRExp_0-0001_resume_1 -c log_line_v2_10b_LRExp_0-0001/checkpoints/CP40.pth
python train.py -l 0.000001 -d 1 -e 40 -b 2 -g --task-type line --data data/dataset_line_v2 --log log_line_v2_10b_LRExp_0-0001_resume_2 -c log_line_v2_10b_LRExp_0-0001_resume_1/checkpoints/CP40.pth
# lr = 0.00001
python train.py -l 0.00001 -d 1 -e 60 -b 2 -g --task-type line --data data/dataset_line_v2 --log log_line_v2_10b_LRExp_0-00001
# lr = 0.000001
python train.py -l 0.000001 -d 1 -e 60 -b 2 -g --task-type line --data data/dataset_line_v2 --log log_line_v2_10b_LRExp_0-000001
# lr = 0.0000001
python train.py -l 0.0000001 -d 1 -e 60 -b 2 -g --task-type line --data data/dataset_line_v2 --log log_line_v2_10b_LRExp_0-0000001