#!/bin/bash

python total_train.py --parallel --batchsize 64 --fp16 --track_mode track2 \
        --train_image_path /home/Bigdata/NICO2/nico/train/ --label2id_path \
        /home/Bigdata/NICO2/ood_label_id_mapping.json --test_image_path /home/Bigdata/NICO2/nico/test/