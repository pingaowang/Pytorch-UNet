#!/usr/bin/env python
# -*- codinbg: utf-8 -*-

python3 train.py -l 0.001 --resize 500 --nz 0 -d 0.99 -e 1000 -b 4 -g --task-type line --data data/dataset_line_v4_val4 --log log_line_v4_val4_id_01
#python3 train.py -l 0.00001 --resize 500 --nz 0 -d 0.99 -e 200 -b 4 -g --task-type line --data data/dataset_line_v4_val4 --log log_line_v4_val4_id_01_resume_1 -c log_line_v4_val4_id_01/checkpoints/CP130.pth
