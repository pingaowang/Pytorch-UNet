#!/usr/bin/env python
# -*- codinbg: utf-8 -*-

# python3 train.py -l 0.0001 --nz 0.001 -d 0.982 -e 300 -b 4 -g --task-type line --data data/dataset_line_v2_b4 --log log_line_v2_full_v3_init
python3 train.py -l 0.00003 --nz 0.001 -d 0.982 -e 300 -b 4 -g --task-type line --data data/dataset_line_v2_b4 --log log_line_v2_full_v3_resume_1 -c log_line_v2_full_v2_init/checkpoints/CP60.pth

