#!/bin/bash

python total_train.py --parallel --batchsize 64 --fp16 --track_mode track1 \
        --train_image_path /home/Bigdata/NICO/nico/train/ --label2id_path \
        /home/Bigdata/NICO/dg_label_id_mapping.json --test_image_path /home/Bigdata/NICO/nico/test/
