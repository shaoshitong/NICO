import os, sys

if __name__ == "__main__":
    import argparse

    paser = argparse.ArgumentParser()
    paser.add_argument("--parallel", default=False, action='store_true')
    paser.add_argument("--cuda_devices", default="0,1,2,3,4,5,6,7",type=str)
    paser.add_argument("--epochs", default=1000000,type=int)
    paser.add_argument("--fp16", default=True, action='store_true')
    paser.add_argument("--train_image_path", default="/home/Bigdata/NICO/nico/train/")
    paser.add_argument("--label2id_path", default="/home/Bigdata/NICO/dg_label_id_mapping.json", type=str)
    paser.add_argument("--test_image_path", default="/home/Bigdata/NICO/nico/test/", type=str)
    args = paser.parse_args()

    stage1 = f"python train.py --batch_size 48 --total_epoch {args.epochs} --lr 0.1 {'--parallel' if args.parallel else ''} {'--fp16' if args.fp16 else ''} " \
             f"--img_size 224 --train_image_path {args.train_image_path} --label2id_path {args.label2id_path} --test_image_path {args.test_image_path}" \
             f" --cuda_devices {args.cuda_devices}"

    stage2 = f"python train.py --batch_size 48 --total_epoch {args.epochs}  --lr 0.1  {'--parallel' if args.parallel else ''} {'--fp16' if args.fp16 else ''} --kd " \
             f"--img_size 224 --train_image_path {args.train_image_path} --label2id_path {args.label2id_path} --test_image_path {args.test_image_path} " \
             f" --cuda_devices {args.cuda_devices}"

    stage3 = f"python train.py --batch_size 16 --total_epoch {args.epochs}  --lr 0.0001  {'--parallel' if args.parallel else ''} {'--fp16' if args.fp16 else ''} " \
             f"--img_size 384 --train_image_path {args.train_image_path} --label2id_path {args.label2id_path} --test_image_path {args.test_image_path} " \
             f"--if_finetune --accumulate_step 4 --cuda_devices {args.cuda_devices} --warmup_epoch -1"

    stage4 = f"python train.py --batch_size 16 --total_epoch {args.epochs} --lr 0.0001  {'--parallel' if args.parallel else ''} {'--fp16' if args.fp16 else ''} --kd " \
             f"--img_size 384 --train_image_path {args.train_image_path} --label2id_path {args.label2id_path} --test_image_path {args.test_image_path} " \
             f"--if_finetune --accumulate_step 4 --cuda_devices {args.cuda_devices} --warmup_epoch -1"

    os.system(stage1)

    os.system('mv original.pth teacher.pth')

    os.system(stage2)

    os.system('mv student.pth resmue.pth')

    os.system(stage3)

    os.system('mv original.pth teacher.pth')

    os.system(stage4)
