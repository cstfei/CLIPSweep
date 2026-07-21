#!/bin/bash

BASE_PATH=/home/user/NLPLab/BadCLIP
# conda activate $BASE_PATH/env

# create badnet
python3 -u backdoor/create_backdoor_data.py --train_data $BASE_PATH/data/CC3M/train.csv --templates data/ImageNet1K/validation/classes.py --num_backdoor 1500 --patch_size 16 --label banana --patch_type random --patch_location random --size_train_data 500000

# create blended
python3 -u backdoor/create_backdoor_data.py --train_data $BASE_PATH/data/CC3M/train.csv --templates data/ImageNet1K/validation/classes.py --size_train_data 500000 --num_backdoor 1500 --label banana --patch_type blended --patch_location blended

# create warped
python3 -u backdoor/create_backdoor_data.py --train_data $BASE_PATH/data/CC3M/train.csv --templates data/ImageNet1K/validation/classes.py --size_train_data 500000 --num_backdoor 1500 --label banana --patch_type warped

# create SIG
python3 -u backdoor/create_backdoor_data.py --train_data $BASE_PATH/data/CC3M/train.csv --templates data/ImageNet1K/validation/classes.py --size_train_data 500000 --num_backdoor 1500 --label banana --patch_type SIG

# clean data 
python3 -u backdoor/create_backdoor_data.py --train_data $BASE_PATH/data/CC3M/train.csv --templates data/ImageNet1K/validation/classes.py --size_train_data 100000 --num_backdoor 0 --label banana --patch_type SIG
