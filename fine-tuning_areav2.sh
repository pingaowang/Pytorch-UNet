#!/usr/bin/env python
# -*- codinbg: utf-8 -*-

#python train.py -l 0.001 -d 0.95 l2 0 -g -e 50 -b 2 --data data/caddata_area_v2_1 --log logs_area_ft/log_area_v2_lr001_l2_0
#python train.py -l 0.01 -d 0.95 l2 0 -g -e 50 -b 2 --data data/caddata_area_v2_1 --log logs_area_ft/log_area_v2_lr01_l2_0
#python train.py -l 0.1 -d 0.95 l2 0 -g -e 50 -b 2 --data data/caddata_area_v2_1 --log logs_area_ft/log_area_v2_lr1e-1_l2_0


#python train.py -l 1 -d 0.95 l2 0.0005 -g -e 50 -b 2 --data data/caddata_area_v2_1 --log logs_area_ft/log_area_v2_lr1e+1_l20005


#python train.py -l 1 -d 0.95 l2 0.005 -g -e 50 -b 2 --data data/caddata_area_v2_1 --log logs_area_ft/log_area_v2_lr1e+1_l2005


#python train.py -l 0.01 -d 0.98 l2 0 --mom 0.9 -g -e 100 -b 2 --data data/caddata_area_v2_1 -c logs_area_ft/log_area_v2_lr01_l2_0/checkpoints/CP50.pth --log logs_area_ft/log_area_v2_l2_0_mom_09
#python train.py -l 0.01 -d 0.98 l2 0.001 --mom 0.9 -g -e 100 -b 2 --data data/caddata_area_v2_1 -c logs_area_ft/log_area_v2_lr01_l2_0/checkpoints/CP50.pth --log logs_area_ft/log_area_v2_lr01_l2_001_mom_09
#python train.py -l 0.01 -d 0.98 l2 0.01 --mom 0.9 -g -e 100 -b 2 --data data/caddata_area_v2_1 -c logs_area_ft/log_area_v2_lr01_l2_0/checkpoints/CP50.pth --log logs_area_ft/log_area_v2_lr01_l2_01_mom_09
#python train.py -l 0.01 -d 0.98 l2 0.1 --mom 0.9 -g -e 100 -b 2 --data data/caddata_area_v2_1 -c logs_area_ft/log_area_v2_lr01_l2_0/checkpoints/CP50.pth --log logs_area_ft/log_area_v2_lr01_l2_1e-1_mom_09
#
#python train.py -l 0.01 -d 0.98 l2 0 --mom 0.99 -g -e 100 -b 2 --data data/caddata_area_v2_1 -c logs_area_ft/log_area_v2_lr01_l2_0/checkpoints/CP50.pth --log logs_area_ft/log_area_v2_l2_0_mom_099
#python train.py -l 0.01 -d 0.98 l2 0.001 --mom 0.99 -g -e 100 -b 2 --data data/caddata_area_v2_1 -c logs_area_ft/log_area_v2_lr01_l2_0/checkpoints/CP50.pth --log logs_area_ft/log_area_v2_lr01_l2_001_mom_099
#python train.py -l 0.01 -d 0.98 l2 0.01 --mom 0.99 -g -e 100 -b 2 --data data/caddata_area_v2_1 -c logs_area_ft/log_area_v2_lr01_l2_0/checkpoints/CP50.pth --log logs_area_ft/log_area_v2_lr01_l2_01_mom_099
#python train.py -l 0.01 -d 0.98 l2 0.1 --mom 0.99 -g -e 100 -b 2 --data data/caddata_area_v2_1 -c logs_area_ft/log_area_v2_lr01_l2_0/checkpoints/CP50.pth --log logs_area_ft/log_area_v2_lr01_l2_1e-1_mom_099



python train.py -l 0.001 -d 0.98 l2 0.01 --mom 0.9 -g -e 1000 -b 2 --data data/caddata_area_v2_1 -c logs_area_ft/log_area_v2_lr01_l2_01_mom_09/checkpoints/CP100.pth --log logs_area_ft/log_area_v2_train


