#!/bin/bash

# Train

cp -r original.pth original.pth

python train.py --batch_size 32 --total_epoch 10000  --lr 0.01  --parallel --fp16 --if_resume --img_size 384 \
        --train_image_path /home/Bigdata/NICO2/nico/train/ --label2id_path /home/Bigdata/NICO2/ood_label_id_mapping.json \
        --test_image_path /home/Bigdata/NICO2/nico/test/ --if_finetune --resume \
        --accumulate_step 4 --cuda_devices 0,1 --warmup_epoch 10 --track_mode track1 --lr_decay_rate 0.

chmod 777 track1_run.sh

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