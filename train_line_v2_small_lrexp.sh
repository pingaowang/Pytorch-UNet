#!/usr/bin/env python
# -*- codinbg: utf-8 -*-

## lr exp: MaxPool, BCELoss, Adam
# lr = 0.1
python3 train.py -l 0.1 -d 1 -e 50 -b 2 -g --task-type line --data data/dataset_line_v2_small --log log_line_v2_10b_LRExp_0-1_BCE
# lr = 0.01
python3 train.py -l 0.01 -d 1 -e 50 -b 2 -g --task-type line --data data/dataset_line_v2_small --log log_line_v2_10b_LRExp_0-01_BCE
# lr = 0.001
python3 train.py -l 0.001 -d 1 -e 50 -b 2 -g --task-type line --data data/dataset_line_v2_small --log log_line_v2_10b_LRExp_0-001_BCE
# lr = 0.0001
python3 train.py -l 0.0001 -d 1 -e 50 -b 2 -g --task-type line --data data/dataset_line_v2_small --log log_line_v2_10b_LRExp_0-0001_BCE
# lr = 0.00001
python3 train.py -l 0.00001 -d 1 -e 50 -b 2 -g --task-type line --data data/dataset_line_v2_small --log log_line_v2_10b_LRExp_0-00001_BCE
# lr = 0.000001
python3 train.py -l 0.000001 -d 1 -e 50 -b 2 -g --task-type line --data data/dataset_line_v2_small --log log_line_v2_10b_LRExp_0-000001_BCE
# lr = 0.0000001
python3 train.py -l 0.0000001 -d 1 -e 50 -b 2 -g --task-type line --data data/dataset_line_v2_small --log log_line_v2_10b_LRExp_0-0000001_BCE
