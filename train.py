import os
import numpy as np
import torchvision
import torch
import torch.nn as nn
from torch.optim import Optimizer
import math
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from models import (DenseNet, pyramidnet272, ConvNeXt, DPN, )
from data.data import get_loader, get_test_loader, write_result
from utils import (EMA,SWA)
from PIL import Image


class KDLoss(nn.KLDivLoss):
    """
    "Distilling the Knowledge in a Neural Network"
    """

    def __init__(self, temperature=4, alpha=1, beta=1, reduction='batchmean', total_epoch=1, **kwargs):
        super().__init__(reduction=reduction)
        self.temperature = temperature
        self.alpha = alpha
        self.beta = 1 - alpha if beta is None else beta
        self.consistency_rampup = total_epoch

    def forward(self, epoch, student_output, teacher_output, targets=None, *args, **kwargs):
        soft_loss = super().forward(torch.log_softmax(student_output / self.temperature, dim=1),
                                    torch.softmax(teacher_output / self.temperature, dim=1))
        if self.alpha is None or self.alpha == 0 or targets is None:
            return soft_loss
        hard_loss = super().forward(torch.log_softmax(student_output, 1), targets)
        return self.alpha * hard_loss + self.beta * (self.temperature ** 2) * soft_loss

    def get_current_consistency_weight(self, epoch):
        return self.sigmoid_rampup(epoch, self.consistency_rampup)

    def sigmoid_rampup(self, current, rampup_length):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = pyramidnet272(num_classes=60)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        x = self.model(x)
        return x

    def load_model(self):
        if os.path.exists('lib/track2_original_epoch_124.pth'):
            start_state = torch.load('lib/track2_original_epoch_124.pth', map_location=self.device)['model']
            self.load_state_dict(start_state)
            print('using teacher loaded model')
            print('-' * 100)

    def save_model(self, name):
        result = self.state_dict()
        torch.save(result, name)


class NoisyStudent():
    def __init__(self,
                 gpu,
                 batch_size: int = 64,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 track_mode='track1',
                 KD=False,
                 ) -> object:
        self.result = {}
        if track_mode == 'track1':
            train_image_path: str = '/root/autodl-tmp/nico/train/'
            valid_image_path: str = '/root/autodl-tmp/nico/train/'
            label2id_path: str = '/root/autodl-tmp/dg_label_id_mapping.json'
            test_image_path: str = '/root/autodl-tmp/nico/test/'
        else:
            train_image_path: str = '/autodl-tmp/nico/train/'
            valid_image_path: str = '/autodl-tmp/nico/train/'
            label2id_path: str = '/home/Bigdata/NICO2/ood_label_id_mapping.json'
            test_image_path: str = '/home/Bigdata/NICO2/nico/test/'
        self.train_loader = get_loader(batch_size=batch_size,
                                       valid_category=None,
                                       train_image_path=train_image_path,
                                       valid_image_path=valid_image_path,
                                       label2id_path=label2id_path,
                                       track_mode=track_mode)
        self.test_loader_predict, _ = get_test_loader(batch_size=batch_size,
                                                      transforms=None,
                                                      label2id_path=label2id_path,
                                                      test_image_path=test_image_path)
        self.test_loader_student, self.label2id = get_test_loader(batch_size=batch_size,
                                                                  transforms='train',
                                                                  label2id_path=label2id_path,
                                                                  test_image_path=test_image_path)
        # self.train_loader = MixLoader([self.train_loader, self.test_loader_student])
        # del self.test_loader_student
        self.gpu = gpu
        self.model = Model().cuda(self.gpu)
        self.KD = KD
        self.lr = lr
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        # self.optimizer = SWA(self.optimizer,swa_start=100*len(self.train_loader), swa_freq=20, swa_lr=0.0005)
        if self.KD:
            self.teacher = Model().cuda(self.gpu)
            self.teacher.load_model()
            self.teacher.train()
            self.teacher.requires_grad_(False)
            self.KDLoss = KDLoss(total_epoch=200)
            self.ema=EMA([self.teacher],[self.model])

    def save_result(self, epoch=None):
        result = {}
        for name, pre in list(self.result.items()):
            _, y = torch.max(pre, dim=1)
            result[name] = y.item()

        if epoch is not None:
            write_result(result, path='prediction' + str(epoch) + '.json')
        else:
            write_result(result)

        return result

    def predict(self):
        with torch.no_grad():
            print('teacher are giving his predictions!')
            self.model.eval()
            for x, names in tqdm(self.test_loader_predict):
                x = x.cuda()
                x = self.model(x)
                for i, name in enumerate(list(names)):
                    self.result[name] = x[i, :].unsqueeze(0)  # 1, D
            print('teacher have given his predictions!')
            print('-' * 100)

    def get_label(self, names):
        y = []
        for name in list(names):
            y.append(self.result[name])

        return torch.tensor(y, device=self.device)

    def train(self,
              total_epoch=3,
              label_smoothing=0.2,
              fp16=True,
              warmup_epoch=1,
              warmup_cycle=12000,
              ):
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
        prev_loss = 999
        train_loss = 99
        criterion = nn.CrossEntropyLoss().cuda(self.gpu)
        dict=torch.load("student_epoch_best.pth")
        # start_epoch=dict['epoch']
        self.model.module.load_state_dict(dict['model'])
        print("successfully load model!!!!!!!!!`")
        # self.optimizer.load_state_dict(dict['optimizer'])
        # scaler.load_state_dict(dict['scaler'])
        # print("start epoch:",start_epoch)
        # # self.model = nn.DataParallel(self.model, device_ids=[0, 1])
        start_epoch=0
        myiter=0
        for epoch in range(start_epoch, total_epoch + 1):
            self.model.train()
            self.warm_up(epoch, now_loss=train_loss, prev_loss=prev_loss)
            prev_loss = train_loss
            train_loss = 0
            train_acc = 0
            step = 0
            pbar = tqdm(self.train_loader)
            for x, y in pbar:
                x = x.cuda(self.gpu)
                y = y.cuda(self.gpu)
                if self.KD:
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
                        nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                        self.optimizer.step()
                    if epoch%50==0:
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
                    train_loss += loss.item()
                    if myiter%2==0:
                        self.optimizer.zero_grad()

                    if fp16:
                        scaler.scale(loss).backward()
                        if myiter%2==1:
                            scaler.unscale_(self.optimizer)
                            nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                            scaler.step(self.optimizer)
                            scaler.update()
                    else:
                        loss.backward()
                        if myiter%2==1:
                            nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                            self.optimizer.step()
                step += 1
                myiter+=1
                if step % 10 == 0:
                    pbar.set_postfix_str(f'loss = {train_loss / step}, acc = {train_acc / step}')

            train_loss /= len(self.train_loader)
            train_acc /= len(self.train_loader)
            if self.gpu == 0:
                print(f'epoch {epoch}, test loader loss = {train_loss}, acc = {train_acc}')
                save_dict = {
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'scaler': scaler.state_dict()
                }
                if self.KD:
                    torch.save(save_dict, 'student_epoch.pth')
                else:
                    torch.save(save_dict, 'original_epoch.pth')
        self.optimizer.swap_swa_sgd()
        self.optimizer.bn_update(self.train_loader,self.model)
        if self.gpu == 0:
            save_dict = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epoch': 200,
                'scaler': scaler.state_dict()
            }
            if self.KD:
                torch.save(save_dict, 'student_epoch_last.pth')
            else:
                torch.save(save_dict, 'original_epoch_last.pth')
    def warm_up(self, epoch, now_loss=None, prev_loss=None):
        if epoch <= 10:
            self.optimizer.param_groups[0]['lr'] = self.lr * epoch / 10
        elif now_loss is not None and prev_loss is not None:
            delta = prev_loss - now_loss
            if delta / now_loss < 0.02 and delta < 0.02:
                self.optimizer.param_groups[0]['lr'] *= 0.8

        p_lr = self.optimizer.param_groups[0]['lr']
        print(f'lr = {p_lr}')

    @torch.no_grad()
    def TTA(self, total_epoch=10, aug_weight=0.5):
        self.predict()
        print('now we are doing TTA')
        for epoch in range(1, total_epoch + 1):
            self.model.eval()
            for x, names in tqdm(self.test_loader_student):
                x = x.cuda(self.gpu)
                x = self.model(x)
                for i, name in enumerate(list(names)):
                    self.result[name] += x[i, :].unsqueeze(0) * aug_weight  # 1, D

        print('TTA finished')
        self.save_result()
        print('-' * 100)


def main_worker(gpu, ngpus_per_node, batch_size, lr, KD, track_mode,total_epoch, dist_url, world_size):
    print("Use GPU: {} for training".format(gpu))
    rank = 0  # 单机
    dist_backend = 'nccl'
    rank = rank * ngpus_per_node + gpu
    dist.init_process_group(backend=dist_backend, init_method=dist_url,
                            world_size=world_size, rank=rank)
    torch.cuda.set_device(gpu)
    batch_size = int(batch_size / ngpus_per_node)
    x = NoisyStudent(gpu=gpu, batch_size=batch_size, lr=lr, KD=KD, track_mode=track_mode)
    x.model = torch.nn.DataParallel(x.model, device_ids=[gpu])
    if KD:
        x.teacher = torch.nn.DataParallel(x.teacher, device_ids=[gpu])
    train_sampler = torch.utils.data.distributed.DistributedSampler(x.train_loader.dataset)
    x.train_loader = torch.utils.data.DataLoader(x.train_loader.dataset, batch_size=batch_size, sampler=train_sampler,
                                                 num_workers=6)
    x.train(total_epoch=total_epoch)
    x.save_result()


if __name__ == '__main__':
    import argparse

    paser = argparse.ArgumentParser()
    paser.add_argument('-b', '--batch_size', default=36)
    paser.add_argument('-t', '--total_epoch', default=100)
    paser.add_argument('-l', '--lr', default=0.0001)
    paser.add_argument('-e', '--test', default=False)
    paser.add_argument('-m', '--mode', default='track1')
    paser.add_argument('-k', '--kd', default=False, type=bool)
    paser.add_argument('-p', '--parallel', default=True, type=bool)
    args = paser.parse_args()
    batch_size = int(args.batch_size)
    total_epoch = int(args.total_epoch)
    track_mode = str(args.mode)
    lr = float(args.lr)
    parallel = args.parallel
    KD = args.kd
    if parallel:
        os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
        world_size = 1
        port_id = 10000 + np.random.randint(0, 1000)
        dist_url = 'tcp://127.0.0.1:' + str(port_id)
        ngpus_per_node = torch.cuda.device_count()
        print('ngpus_per_node', ngpus_per_node)
        world_size = ngpus_per_node * world_size
        print('multiprocessing_distributed')
        torch.multiprocessing.set_start_method('spawn')
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, batch_size, lr, KD, track_mode, total_epoch, dist_url, world_size))
    else:
        if args.test:
            x = NoisyStudent(gpu=0, batch_size=batch_size, lr=lr, track_mode=track_mode)
            x.predict()
            x.save_result()
        else:
            x = NoisyStudent(gpu=0, batch_size=batch_size, lr=lr, KD=KD, track_mode=track_mode)
            x.train(total_epoch=total_epoch)
            x.save_result()
