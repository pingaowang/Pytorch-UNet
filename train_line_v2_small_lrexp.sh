#!/usr/bin/env python
# -*- codinbg: utf-8 -*-

## lr exp: MaxPool, BCELoss, Adam
# lr = 0.1
python train.py -l 0.1 -d 1 -e 40 -b 2 -g --task-type line --data data/dataset_line_v2_small --log log_line_v2_10b_LRExp_0-1
# lr = 0.01
python train.py -l 0.01 -d 1 -e 40 -b 2 -g --task-type line --data data/dataset_line_v2_small --log log_line_v2_10b_LRExp_0-01
# lr = 0.001
python train.py -l 0.001 -d 1 -e 40 -b 2 -g --task-type line --data data/dataset_line_v2_small --log log_line_v2_10b_LRExp_0-001
# lr = 0.0001
python train.py -l 0.0001 -d 1 -e 40 -b 2 -g --task-type line --data data/dataset_line_v2_small --log log_line_v2_10b_LRExp_0-0001
# lr = 0.00001
python train.py -l 0.00001 -d 1 -e 60 -b 2 -g --task-type line --data data/dataset_line_v2_small --log log_line_v2_10b_LRExp_0-00001
# lr = 0.000001
python train.py -l 0.000001 -d 1 -e 60 -b 2 -g --task-type line --data data/dataset_line_v2_small --log log_line_v2_10b_LRExp_0-000001
# lr = 0.0000001
python train.py -l 0.0000001 -d 1 -e 60 -b 2 -g --task-type line --data data/dataset_line_v2_small --log log_line_v2_10b_LRExp_0-0000001