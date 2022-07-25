import os

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.utils.data.distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from data import cutmix, get_test_loader, get_train_loader, mixup, write_result
from models import pyramidnet272
from utils import EMA, SWA


class KDLoss(nn.KLDivLoss):
    """
    "Distilling the Knowledge in a Neural Network"
    """

    def __init__(self, temperature=5, alpha=1, beta=0.2, reduction="batchmean", **kwargs):
        super().__init__(reduction=reduction)
        self.temperature = temperature
        self.alpha = alpha
        self.beta = 1 - alpha if beta is None else beta

    def forward(self, epoch, student_output, teacher_output, targets=None, *args, **kwargs):
        soft_loss = super().forward(
            torch.log_softmax(student_output / self.temperature, dim=1),
            torch.softmax(teacher_output / self.temperature, dim=1),
        )
        hard_loss = super().forward(torch.log_softmax(student_output, 1), targets)
        return self.alpha * hard_loss + self.beta * (self.temperature ** 2) * soft_loss


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


class NoisyStudent:
    def __init__(
        self,
        gpu,
        train_image_path: str = "/home/Bigdata/NICO/nico/train/",
        label2id_path: str = "/home/Bigdata/NICO/dg_label_id_mapping.json",
        test_image_path: str = "/home/Bigdata/NICO/nico/test/",
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_epoch=10,
        track_mode="track1",
        kd=False,
        teacher_ckpt_path="./teacher.pth",
        student_ckpt_path="./student.pth",
        original_ckpt_path="./original.pth",
        ensemble=False,
        img_size=224,
        cutmix_in_cpu=False,
        if_finetune=False,
        parallel=False,
    ):
        self.result = {}
        train_image_path: str = train_image_path
        label2id_path: str = label2id_path
        test_image_path: str = test_image_path
        self.train_loader = get_train_loader(
            batch_size=batch_size,
            train_image_path=train_image_path,
            label2id_path=label2id_path,
            track_mode=track_mode,
            img_size=img_size,
            cutmix_in_cpu=cutmix_in_cpu,
        )
        self.test_loader_predict, _ = get_test_loader(
            batch_size=batch_size,
            transforms=None,
            label2id_path=label2id_path,
            test_image_path=test_image_path,
            img_size=img_size,
            track_mode=track_mode,
            cutmix_in_cpu=cutmix_in_cpu,
        )
        self.test_loader_student, self.label2id = get_test_loader(
            batch_size=batch_size,
            transforms="train",
            label2id_path=label2id_path,
            test_image_path=test_image_path,
            track_mode=track_mode,
            cutmix_in_cpu=cutmix_in_cpu,
        )
        self.gpu = gpu
        self.warmup_epoch = warmup_epoch
        self.kd = kd
        lr if not if_finetune else min(lr, 1e-4)
        self.lr = lr
        self.parallel = parallel
        self.cutmix_in_cpu = cutmix_in_cpu
        self.teacher_ckpt_path = teacher_ckpt_path
        self.student_ckpt_path = student_ckpt_path
        self.original_ckpt_path = original_ckpt_path
        self.model = pyramidnet272(num_classes=60, num_models=2 if ensemble else -1).cuda(self.gpu)
        if if_finetune:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
            )
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
            )
        if self.kd:
            self.teacher = pyramidnet272(num_classes=60, num_models=2 if ensemble else -1).cuda(
                self.gpu
            )
            dict = torch.load(self.teacher_ckpt_path)
            self.teacher.load_state_dict(dict["model"])
            self.model.load_state_dict(dict["model"])
            self.teacher.eval()
            self.teacher.requires_grad_(False)
            print("use knowledge distillation...")
            self.KDLoss = KDLoss()
            self.ema = EMA([self.teacher], [self.model], momentum=0.99)

    def save_result(self, epoch=None):
        result = {}
        for name, pre in list(self.result.items()):
            _, y = torch.max(pre, dim=1)
            result[name] = y.item()

        if epoch is not None:
            write_result(result, path="prediction" + str(epoch) + ".json")
        else:
            write_result(result)

        return result

    def predict(self):
        with torch.no_grad():
            print("teacher are giving his predictions!")
            self.model.eval()
            for x, names in tqdm(self.test_loader_predict):
                x = x.cuda()
                x = self.model(x)
                for i, name in enumerate(list(names)):
                    self.result[name] = x[i, :].unsqueeze(0)  # 1, D
            print("teacher have given his predictions!")
            print("-" * 100)

    def get_label(self, names):
        y = []
        for name in list(names):
            y.append(self.result[name])

        return torch.tensor(y).cuda(self.gpu)

    def train(
        self,
        total_epoch,
        accumulate_step=1,
        decay_rate=0.9,
        fp16=True,
        if_resmue=False,
    ):
        from torch.cuda.amp import GradScaler, autocast

        scaler = GradScaler()
        prev_loss = 999
        train_loss = 99
        criterion = nn.CrossEntropyLoss().cuda(self.gpu)
        start_epoch = 1
        myiter = 0
        min_lr = max(self.lr * 0.0001, 1e-6)
        if if_resmue:
            model_state_dict = torch.load("resume.pth")["model"]
            if self.parallel:
                self.model.module.load_state_dict(model_state_dict)
            else:
                self.model.load_state_dict(model_state_dict)
            print("successfully load 224x224 model's ckpt file")
        for epoch in range(start_epoch, total_epoch + 1):
            if self.optimizer.param_groups[0]["lr"] < min_lr and epoch > self.warmup_epoch + 1:
                break
            self.model.train()
            self.warm_up(epoch, now_loss=train_loss, prev_loss=prev_loss, decay_rate=decay_rate)
            prev_loss = train_loss
            train_loss = 0
            train_acc = 0
            step = 0
            pbar = tqdm(self.train_loader)
            for x, y in pbar:
                x = x.cuda(self.gpu)
                y = y.cuda(self.gpu)
                if self.cutmix_in_cpu == False:
                    x, y = cutmix(x, y)
                if self.kd:
                    if fp16:
                        with autocast():
                            sx = self.model(x)  # N, 60
                            with torch.no_grad():
                                tx = self.teacher(x)
                            _, pre = torch.max(sx, dim=1)
                            loss = self.KDLoss(epoch, sx, tx, y)
                    else:
                        sx = self.model(x)  # N, 60
                        tx = self.teacher(x)
                        _, pre = torch.max(sx, dim=1)
                        loss = self.KDLoss(epoch, sx, tx, y)
                    if pre.shape != y.shape:
                        _, y = torch.max(y, dim=1)
                    train_acc += (torch.sum(pre == y).item()) / y.shape[0]
                    if self.parallel:
                        train_loss += reduce_mean(loss, torch.cuda.device_count()).item()
                    else:
                        train_loss += loss.item()
                    self.optimizer.zero_grad()
                    if fp16:
                        scaler.scale(loss).backward()
                        scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()
                    self.ema.step()
                else:
                    if fp16:
                        with autocast():
                            x = self.model(x)  # N, 60
                            _, pre = torch.max(x, dim=1)
                            loss = criterion(x, y)
                    else:
                        x = self.model(x)  # N, 60
                        _, pre = torch.max(x, dim=1)
                        loss = criterion(x, y)
                    if pre.shape != y.shape:
                        _, y = torch.max(y, dim=1)

                    train_acc += (torch.sum(pre == y).item()) / y.shape[0]
                    if self.parallel:
                        train_loss += reduce_mean(loss, torch.cuda.device_count()).item()
                    else:
                        train_loss += loss.item()

                    if myiter % accumulate_step == 0:
                        self.optimizer.zero_grad()

                    if fp16:
                        scaler.scale(loss).backward()
                        if myiter % accumulate_step == accumulate_step - 1:
                            scaler.unscale_(self.optimizer)
                            nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                            scaler.step(self.optimizer)
                            scaler.update()
                    else:
                        loss.backward()
                        if myiter % accumulate_step == accumulate_step - 1:
                            self.optimizer.step()
                step += 1
                myiter += 1
                if step % 10 == 0:
                    pbar.set_postfix_str(f"loss = {train_loss / step}, acc = {train_acc / step}")
            train_loss /= len(self.train_loader)
            train_acc /= len(self.train_loader)
            if self.gpu == 0:
                print(f"epoch {epoch}, test loader loss = {train_loss}, acc = {train_acc}")
                now_model = self.model if not self.parallel else self.model.module
                save_dict = {
                    "model": now_model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "scaler": scaler.state_dict(),
                }
                if self.kd:
                    torch.save(save_dict, self.student_ckpt_path)
                else:
                    torch.save(save_dict, self.original_ckpt_path)

    def warm_up(self, epoch, now_loss=None, prev_loss=None, decay_rate=0.9):
        if epoch <= self.warmup_epoch:
            self.optimizer.param_groups[0]["lr"] = self.lr * epoch / self.warmup_epoch
        elif now_loss is not None and prev_loss is not None:
            delta = prev_loss - now_loss
            if delta / now_loss < 0.02 and delta < 0.02 and delta > -0.02:
                self.optimizer.param_groups[0]["lr"] *= decay_rate
        p_lr = self.optimizer.param_groups[0]["lr"]
        print(f"lr = {p_lr}")

    @torch.no_grad()
    def TTA(self, total_epoch=100, aug_weight=0.5):
        self.predict()
        print("now we are doing TTA")
        for epoch in range(1, total_epoch + 1):
            self.model.eval()
            for x, names in tqdm(self.test_loader_student):
                x = x.cuda(self.gpu)
                x = self.model(x)
                for i, name in enumerate(list(names)):
                    self.result[name] += x[i, :].unsqueeze(0) * aug_weight
        print("TTA finished")
        self.save_result()


def main_worker(
    gpu,
    ngpus_per_node,
    batch_size,
    lr,
    kd,
    total_epoch,
    dist_url,
    world_size,
    ensemble,
    warmup_epoch,
    train_image_path,
    label2id_path,
    test_image_path,
    img_size,
    cutmix_in_cpu,
    lr_decay_rate,
    fp16,
    if_finetune,
    accumulate_step,
    if_resume,
    track_mode,
):
    print("Use GPU: {} for training".format(gpu))
    rank = 0  # 单机
    dist_backend = "nccl"
    rank = rank * ngpus_per_node + gpu
    print("world_size:", world_size)
    dist.init_process_group(
        backend=dist_backend, init_method=dist_url, world_size=world_size, rank=rank
    )
    torch.cuda.set_device(gpu)
    batch_size = int(batch_size / ngpus_per_node)
    print("sub batch size is", batch_size)
    x = NoisyStudent(
        gpu=gpu,
        batch_size=batch_size,
        kd=kd,
        lr=lr,
        warmup_epoch=warmup_epoch,
        ensemble=ensemble,
        train_image_path=train_image_path,
        label2id_path=label2id_path,
        test_image_path=test_image_path,
        img_size=img_size,
        cutmix_in_cpu=cutmix_in_cpu,
        if_finetune=if_finetune,
        track_mode=track_mode,
        parallel=True,
    )
    x.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(x.model)
    x.model = DDP(x.model, device_ids=[gpu], output_device=gpu)
    if kd:
        x.teacher = torch.nn.DataParallel(x.teacher, device_ids=[gpu], output_device=gpu)
    train_sampler = torch.utils.data.distributed.DistributedSampler(x.train_loader.dataset)
    x.train_loader = torch.utils.data.DataLoader(
        x.train_loader.dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=6 if batch_size % 6 == 0 else 4,
        pin_memory=True,
    )
    x.train(
        total_epoch=total_epoch,
        decay_rate=lr_decay_rate,
        fp16=fp16,
        accumulate_step=accumulate_step,
        if_resmue=if_resume,
    )


if __name__ == "__main__":
    import argparse

    paser = argparse.ArgumentParser()
    paser.add_argument("--batch_size", default=10, type=int)
    paser.add_argument("--warmup_epoch", default=10, type=int)
    paser.add_argument("--total_epoch", default=200, type=int)
    paser.add_argument("--lr", default=0.1, type=float)
    paser.add_argument("--test", default=False, action="store_true")
    paser.add_argument("--kd", default=False, action="store_true")
    paser.add_argument("--parallel", default=False, action="store_true")
    paser.add_argument("--ensemble", default=False, action="store_true")
    paser.add_argument("--img_size", default=224, type=int)
    paser.add_argument("--cuda_devices", default="0,1,2,3", type=str)
    paser.add_argument("--cutmix_in_cpu", default=False, action="store_true")
    paser.add_argument("--fp16", default=False, action="store_true")
    paser.add_argument("--lr_decay_rate", default=0.9, type=float)
    paser.add_argument("--accumulate_step", default=1, type=int)
    paser.add_argument("--if_finetune", default=False, action="store_true")
    paser.add_argument("--if_resume", default=False, action="store_true")
    paser.add_argument("--track_mode", default="track1", type=str)
    paser.add_argument("--train_image_path", default="/home/Bigdata/NICO/nico/train/", type=str)
    paser.add_argument(
        "--label2id_path", default="/home/Bigdata/NICO/dg_label_id_mapping.json", type=str
    )
    paser.add_argument("--test_image_path", default="/home/Bigdata/NICO/nico/test/", type=str)
    args = paser.parse_args()

    print(args)
    batch_size = args.batch_size
    total_epoch = args.total_epoch
    lr = args.lr
    ensemble = args.ensemble
    parallel = args.parallel
    kd = args.kd
    if parallel and args.test == False:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
        os.environ["MASTER_ADDR"] = "127.0.0.1"  #
        os.environ["MASTER_PORT"] = "8889"  #
        world_size = 1
        port_id = 10000 + np.random.randint(0, 1000)
        dist_url = "tcp://127.0.0.1:" + str(port_id)
        ngpus_per_node = torch.cuda.device_count()
        print("ngpus_per_node", ngpus_per_node)
        world_size = ngpus_per_node * world_size
        print("multiprocessing_distributed")
        torch.multiprocessing.set_start_method("spawn")
        mp.spawn(
            main_worker,
            nprocs=ngpus_per_node,
            args=(
                ngpus_per_node,
                batch_size,
                lr,
                kd,
                total_epoch,
                dist_url,
                world_size,
                ensemble,
                args.warmup_epoch,
                args.train_image_path,
                args.label2id_path,
                args.test_image_path,
                args.img_size,
                args.cutmix_in_cpu,
                args.lr_decay_rate,
                args.fp16,
                args.if_finetune,
                args.accumulate_step,
                args.if_resume,
                args.track_mode,
            ),
        )
    else:
        if args.test:
            x = NoisyStudent(
                gpu=0,
                batch_size=batch_size,
                kd=kd,
                lr=lr,
                warmup_epoch=args.warmup_epoch,
                ensemble=ensemble,
                train_image_path=args.train_image_path,
                label2id_path=args.label2id_path,
                test_image_path=args.test_image_path,
                img_size=args.img_size,
                cutmix_in_cpu=args.cutmix_in_cpu,
                if_finetune=args.if_finetune,
                track_mode=args.track_mode,
                parallel=False,
            )
            x.model.load_state_dict(torch.load("resume.pth")['model'])
            x.TTA()
            x.save_result()
        else:
            x = NoisyStudent(
                gpu=0,
                batch_size=batch_size,
                kd=kd,
                lr=lr,
                warmup_epoch=args.warmup_epoch,
                ensemble=ensemble,
                train_image_path=args.train_image_path,
                label2id_path=args.label2id_path,
                test_image_path=args.test_image_path,
                img_size=args.img_size,
                cutmix_in_cpu=args.cutmix_in_cpu,
                if_finetune=args.if_finetune,
                track_mode=args.track_mode,
                parallel=False,
            )
            x.train(
                total_epoch=total_epoch,
                decay_rate=args.lr_decay_rate,
                fp16=args.fp16,
                accumulate_step=args.accumulate_step,
                if_resmue=args.if_resume,
            )
