#!/usr/bin/env python
# -*- codinbg: utf-8 -*-


#python3 train.py -l 0.0001 --resize 200 --nz 0.001 -d 1 -e 60 -b 24 -g --task-type line --data data/dataset_line_v3_val4 --log log_line_v3_val4_size-200_nz0-001
#python3 train.py -l 0.0001 --resize 200 --nz 0.0001 -d 1 -e 60 -b 24 -g --task-type line --data data/dataset_line_v3_val4 --log log_line_v3_val4_size-200_nz0-0001
#python3 train.py -l 0.0001 --resize 200 --nz 0.00001 -d 1 -e 60 -b 24 -g --task-type line --data data/dataset_line_v3_val4 --log log_line_v3_val4_size-200_nz0-00001
#python3 train.py -l 0.0001 --resize 200 --nz 0 -d 1 -e 60 -b 24 -g --task-type line --data data/dataset_line_v3_val4 --log log_line_v3_val4_size-200_nz0

python3 train.py -l 0.0001 --resize 500 --nz 0.001 -d 1 -e 60 -b 4 -g --task-type line --data data/dataset_line_v3_val4 --log log_line_v3_val4_size-500_nz0-001
python3 train.py -l 0.0001 --resize 500 --nz 0.0001 -d 1 -e 60 -b 4 -g --task-type line --data data/dataset_line_v3_val4 --log log_line_v3_val4_size-500_nz0-0001
python3 train.py -l 0.0001 --resize 500 --nz 0.00001 -d 1 -e 60 -b 4 -g --task-type line --data data/dataset_line_v3_val4 --log log_line_v3_val4_size-500_nz0-00001
python3 train.py -l 0.0001 --resize 500 --nz 0 -d 1 -e 60 -b 4 -g --task-type line --data data/dataset_line_v3_val4 --log log_line_v3_val4_size-500_nz0