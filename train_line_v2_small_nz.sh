#!/usr/bin/env python
# -*- codinbg: utf-8 -*-

## line_v2_small dataset, + loss_non_zero = 1
python3 train.py -l 0.0001 -nz 1 -d 1 -e 40 -b 4 -g --task-type line --data data/dataset_line_v2_small --log log_line_v2_10b_nz

## line_v2_small dataset, + loss_non_zero = 0.5
python3 train.py -l 0.0001 -nz 0.5 -d 1 -e 40 -b 4 -g --task-type line --data data/dataset_line_v2_small --log log_line_v2_10b_nz

## line_v2_small dataset, + loss_non_zero = 0.1
python3 train.py -l 0.0001 -nz 0.1 -d 1 -e 40 -b 4 -g --task-type line --data data/dataset_line_v2_small --log log_line_v2_10b_nz

## line_v2_small dataset, + loss_non_zero = 0.01
python3 train.py -l 0.0001 -nz 0.01 -d 1 -e 40 -b 4 -g --task-type line --data data/dataset_line_v2_small --log log_line_v2_10b_nz