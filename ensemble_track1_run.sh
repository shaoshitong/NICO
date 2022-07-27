#!/bin/bash

# Train

cp -r original.pth original_tmp.pth

cp -r resume.pth resume_tmp.pth
python train.py --batch_size 64 --total_epoch 10000  --lr 0.1  --parallel --fp16 --kd --img_size 224 \
        --train_image_path /home/Bigdata/NICO2/nico/train/ --label2id_path /home/Bigdata/NICO2/dg_label_id_mapping.json \
        --test_image_path /home/Bigdata/NICO2/nico/test/ --lr_decay_rate 0.95 \
        --cuda_devices 0,1 --warmup_epoch 10 --track_mode track1

mv student.pth resume.pth

chmod 777 track1_run.sh

bash track1_run.sh

mv original.pth original_1.pth

bash track1_run.sh

mv original.pth original_2.pth

bash track1_run.sh

mv original.pth original_3.pth



# Test

#mkdir prediction
#
#python train.py --test --test_pth_path original_1.pth --batch_size 48 --img_size 384 --cutmix_in_cpu --track_mode track1 \
#        --train_image_path /home/Bigdata/NICO2/nico/train/ --label2id_path \
#        /home/Bigdata/NICO2/ood_label_id_mapping.json --test_image_path /home/Bigdata/NICO2/nico/test/
#
#mv prediction.zip  prediction/prediction_0.zip
#
#python train.py --test --test_pth_path original_2.pth --batch_size 48 --img_size 384 --cutmix_in_cpu --track_mode track1 \
#        --train_image_path /home/Bigdata/NICO2/nico/train/ --label2id_path \
#        /home/Bigdata/NICO2/ood_label_id_mapping.json --test_image_path /home/Bigdata/NICO2/nico/test/
#
#mv prediction.zip  prediction/prediction_1.zip
#
#python train.py --test --test_pth_path original_3.pth --batch_size 48 --img_size 384 --cutmix_in_cpu --track_mode track1 \
#        --train_image_path /home/Bigdata/NICO2/nico/train/ --label2id_path \
#        /home/Bigdata/NICO2/ood_label_id_mapping.json --test_image_path /home/Bigdata/NICO2/nico/test/
#
#mv prediction.zip  prediction/prediction_2.zip