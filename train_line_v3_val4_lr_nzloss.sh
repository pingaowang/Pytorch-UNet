#!/usr/bin/env python
# -*- codinbg: utf-8 -*-


#python3 train.py -l 0.0001 --resize 200 --nz 0.001 -d 0.982 -e 200 -b 24 -g --task-type line --data data/dataset_line_v3_val4 --log log_line_v3_val4_lr0-0001_nz0-001
python3 train.py -l 0.0001 --resize 200 --nz 0.0001 -d 0.982 -e 200 -b 24 -g --task-type line --data data/dataset_line_v3_val4 --log log_line_v3_val4_lr0-0001_nz0-0001
python3 train.py -l 0.0001 --resize 200 --nz 0.00001 -d 0.982 -e 200 -b 24 -g --task-type line --data data/dataset_line_v3_val4 --log log_line_v3_val4_lr0-0001_nz0-00001
python3 train.py -l 0.0001 --resize 200 --nz 0 -d 0.982 -e 200 -b 24 -g --task-type line --data data/dataset_line_v3_val4 --log log_line_v3_val4_lr0-0001_nz0
