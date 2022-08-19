import os

import torch

if __name__ == "__main__":
    import argparse

    paser = argparse.ArgumentParser()
    paser.add_argument(
        "--parallel",
        default=False,
        action="store_true",
        help="whether to use single multi-card mode",
    )
    paser.add_argument(
        "--cuda_devices",
        default="0,1",
        type=str,
        help="if using single multi-card mode, what is the GPU number used",
    )
    paser.add_argument(
        "--epochs",
        default=1000000,
        type=int,
        help="the total number of training epochs. Since the code adds a setting to stop training when the learning rate is less than a certain value, this hyperparameter can be set to infinity",
    )
    paser.add_argument(
        "--track_mode",
        default="track1",
        type=str,
        help="track mode should be one of track1 and track2",
    )
    paser.add_argument(
        "--batchsize",
        default=48,
        type=int,
        help="batch size of training dataset and testing dataset",
    )
    paser.add_argument("--fp16", default=False, action="store_true", help="if use amp")
    paser.add_argument(
        "--train_image_path",
        default="./nico/train/",
        help="location of the images to be trained",
    )
    paser.add_argument(
        "--label2id_path",
        default="./nico/dg_label_id_mapping.json",
        type=str,
        help="location of the labels' name",
    )
    paser.add_argument(
        "--test_image_path",
        default="./nico/test/",
        type=str,
        help="location of the images to be tested",
    )
    args = paser.parse_args()

    assert args.track_mode in ["track1", "track2"], "track mode should be one of track1 and track2!"
    stage1 = (
        f"python train.py --batch_size {args.batchsize} --total_epoch {args.epochs} --lr 0.1 {'--parallel' if args.parallel else ''} {'--fp16' if args.fp16 else ''} "
        f"--img_size 224 --train_image_path {args.train_image_path} --label2id_path {args.label2id_path} --test_image_path {args.test_image_path}"
        f" --cuda_devices {args.cuda_devices} --track_mode {args.track_mode} --lr_decay_rate 0.9"
    )

    stage2 = (
        f"python train.py --batch_size {args.batchsize} --total_epoch {args.epochs}  --lr 0.1  {'--parallel' if args.parallel else ''} {'--fp16' if args.fp16 else ''} --kd "
        f"--img_size 224 --train_image_path {args.train_image_path} --label2id_path {args.label2id_path} --test_image_path {args.test_image_path} "
        f" --cuda_devices {args.cuda_devices} --track_mode {args.track_mode} --lr_decay_rate 0.9"
    )

    stage3 = (
        f"python train.py --batch_size {int(args.batchsize/2)} --total_epoch {args.epochs}  --lr 0.01  {'--parallel' if args.parallel else ''} {'--fp16' if args.fp16 else ''} --if_resume "
        f"--img_size 384 --train_image_path {args.train_image_path} --label2id_path {args.label2id_path} --test_image_path {args.test_image_path} "
        f"--if_finetune --accumulate_step 4 --cuda_devices {args.cuda_devices} --warmup_epoch 10 --track_mode {args.track_mode} --lr_decay_rate 0.8"
    )

    stage4 = (
        f"python train.py --batch_size {int(args.batchsize/2)} --total_epoch {args.epochs} --lr 0.01  {'--parallel' if args.parallel else ''} {'--fp16' if args.fp16 else ''} --kd "
        f"--img_size 384 --train_image_path {args.train_image_path} --label2id_path {args.label2id_path} --test_image_path {args.test_image_path} "
        f"--if_finetune --accumulate_step 4 --cuda_devices {args.cuda_devices} --warmup_epoch 10 --track_mode {args.track_mode} --lr_decay_rate 0.8"
    )

    os.system(stage1)

    os.system("mv original.pth teacher.pth")

    torch.cuda.empty_cache()

    print("=" * 60 + "compete stage1" + "=" * 60)

    os.system(stage2)

    os.system("mv student.pth resume.pth")

    torch.cuda.empty_cache()

    print("=" * 60 + "compete stage2" + "=" * 60)

    os.system(stage3)

    os.system("mv original.pth teacher.pth")

    torch.cuda.empty_cache()

    print("=" * 60 + "compete stage3" + "=" * 60)

    os.system(stage4)

    print("=" * 60 + "compete stage4" + "=" * 60)
