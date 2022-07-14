#!/bin/bash

mkdir prediction

chmod 777 track1_run.sh

bash track1_run.sh

python train.py --test --batch_size 48 --img_size 384 --cutmix_in_cpu --track_mode track1 \
        --train_image_path /home/Bigdata/NICO2/nico/train/ --label2id_path \
        /home/Bigdata/NICO2/ood_label_id_mapping.json --test_image_path /home/Bigdata/NICO2/nico/test/

mv prediction.json prediction/prediction_1.json

bash track1_run.sh

python train.py --test --batch_size 48 --img_size 384 --cutmix_in_cpu --track_mode track1 \
        --train_image_path /home/Bigdata/NICO2/nico/train/ --label2id_path \
        /home/Bigdata/NICO2/ood_label_id_mapping.json --test_image_path /home/Bigdata/NICO2/nico/test/

mv prediction.json prediction/prediction_2.json

bash track1_run.sh

python train.py --test --batch_size 48 --img_size 384 --cutmix_in_cpu --track_mode track1 \
        --train_image_path /home/Bigdata/NICO2/nico/train/ --label2id_path \
        /home/Bigdata/NICO2/ood_label_id_mapping.json --test_image_path /home/Bigdata/NICO2/nico/test/

mv prediction.json prediction/prediction_3.json


bash track1_run.sh

python train.py --test --batch_size 48 --img_size 384 --cutmix_in_cpu --track_mode track1 \
        --train_image_path /home/Bigdata/NICO2/nico/train/ --label2id_path \
        /home/Bigdata/NICO2/ood_label_id_mapping.json --test_image_path /home/Bigdata/NICO2/nico/test/

mv prediction.json prediction/prediction_4.json

bash track1_run.sh

python train.py --test --batch_size 48 --img_size 384 --cutmix_in_cpu --track_mode track1 \
        --train_image_path /home/Bigdata/NICO2/nico/train/ --label2id_path \
        /home/Bigdata/NICO2/ood_label_id_mapping.json --test_image_path /home/Bigdata/NICO2/nico/test/

mv prediction.json prediction/prediction_5.json

python ensemble.py