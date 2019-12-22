#!/usr/bin/env python
# -*- codinbg: utf-8 -*-

python3 train.py -l 0.0001 --resize 500 --nz 0 -d 0.9 -e 100 -b 4 -g --task-type line --data data/dataset_line_v4_val4 --log log_line_v4_val4_id_03
#python3 train.py -l 0.00001 --resize 200 --nz 0 -d 0.95 -e 200 -b 24 -g --task-type line --data data/dataset_line_v4_val4 --log log_line_v4_val4_id_03_resume1 -c log_line_v4_val4_id_03/checkpoints/CP100.pth
