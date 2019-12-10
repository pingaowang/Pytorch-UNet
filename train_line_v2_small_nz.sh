#!/usr/bin/env python
# -*- codinbg: utf-8 -*-

## line_v2_small dataset, + loss_non_zero
python train.py -l 0.0001 -d 1 -e 40 -b 2 -g --task-type line --data data/dataset_line_v2_small --log log_line_v2_10b_nz