import torch
import model
import helper
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from othermodel.pyramidnet import get_pyramidnet
from torch.utils.data import SubsetRandomSampler
import os
import shutil
import argparse
import numpy as np
import torchvision
import torchvision.transforms as transforms
from bisect import bisect_right
import time
import math
import os
helper.fix_seed(0)
scaler=torch.cuda.amp.GradScaler()
import torch
class KDLoss(nn.KLDivLoss):
    """
    "Distilling the Knowledge in a Neural Network"
    """
    def __init__(self, temperature, alpha=1, beta=1, reduction='batchmean', **kwargs):
        super().__init__(reduction=reduction)
        self.temperature = temperature
        self.alpha = alpha
        self.beta = 1 - alpha if beta is None else beta
    def forward(self, student_output, teacher_output, targets=None, *args, **kwargs):
        soft_loss = super().forward(torch.log_softmax(student_output / self.temperature, dim=1),
                                    torch.softmax(teacher_output / self.temperature, dim=1))
        if self.alpha is None or self.alpha == 0 or targets is None:
            return soft_loss
        hard_loss = super().forward(torch.log_softmax(student_output,1),targets)
        return self.alpha * hard_loss + self.beta * (self.temperature ** 2) * soft_loss


def main_worker(gpu, ngpus_per_node,rank,world_size,dist_url,args):
    rank = rank * ngpus_per_node + gpu
    dist.init_process_group(backend='nccl', init_method=dist_url,
                        world_size=world_size, rank=rank)
    net = model.PyramidNet(num_classes=60,args=args,shakedrop=True)
    net.cuda(gpu)
    torch.backends.cudnn.benchmark=True
    torch.backends.cudnn.fastest=True
    net.network = torch.nn.parallel.DistributedDataParallel(net.network, device_ids=[gpu])
    args.batch_size = int(args.batch_size / ngpus_per_node)
    train_dataloader,test_dataloader = helper.get_our_dataloader(args,'/home/sst/dataset/nico/nico/train' ,'train')
    print("begin training......")
    if args.resume==True:
        dict=torch.load("./run2.pth")
        epoch=dict['epoch']
        net.optimizer.load_state_dict(dict['optimizer'])
        net.network.module.load_state_dict(dict['model'])
    else:
        epoch=0
    best_acc=0.
    if args.KD==True:
        kd_loss=KDLoss(temperature=4)
        tnet=model.PyramidNet(args.num_classes,args,blocks=200,alpha=360)
        tnet.cuda(gpu)
        tnet.network=torch.nn.parallel.DistributedDataParallel(tnet.network,device_ids=[gpu])
        model_ckpt=torch.load("./results/kd_best4.pth")['model']
        tnet.network.module.load_state_dict(model_ckpt)
        tnet.requires_grad=False
        for step in range(epoch,args.num_steps):
            eps=(step+1)/args.num_steps
            train_dataloader.dataset.eps=eps
            test_dataloader.dataset.eps=eps
            if gpu==0:
                print(f"lr:{net.optimizer.param_groups[0]['lr']}")
            for i,(x,y,domain,real_domain) in enumerate(train_dataloader):
                net.train()
                x, y,domain,real_domain= x.cuda(gpu), y.cuda(gpu),domain.cuda(gpu),real_domain.cuda(gpu)
                mini_batches=[x,y,domain,real_domain]
                if args.amp:
                    with torch.cuda.amp.autocast(enabled=True):
                        all_logits = net.predict(x)
                    with torch.no_grad():
                        t_all_logits=tnet.predict(x)
                    with torch.cuda.amp.autocast(enabled=True):
                        loss=kd_loss(all_logits,t_all_logits,y)
                    net.optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(net.optimizer)
                    scaler.update()
                    step_vals ={'kd_loss': loss.item()}
                else:
                    step_vals = net.update(mini_batches)
                if (i + 1) % 100 == 0 and gpu == 0:
                    log_str = "Step %d " % (step + 1)
                    for k, v in step_vals.items():
                        log_str = log_str + "%s: %.4f, " % (k, v)
                    print(log_str)
                    dict = {'epoch': step, 'model': net.network.module.state_dict(),
                            'optimizer': net.optimizer.state_dict()}
                    torch.save(dict, "run2.pth")
            net.scheduler.step(step)
            if (step + 1) % 1 == 0 and gpu == 0:
                accuracy = helper.test(net, test_dataloader, gpu)
                if accuracy > best_acc:
                    best_acc = accuracy
                    dict = {'epoch': step, 'model': net.network.module.state_dict(),
                            'optimizer': net.optimizer.state_dict()}
                    torch.save(dict, "results/kd_best5.pth")
                print("ite: %d, test accuracy: %.4f" % (step + 1, accuracy))

    else:
        for step in range(epoch,args.num_steps):
            if gpu==0:
                print(f"lr:{net.optimizer.param_groups[0]['lr']}")
            for i,(x,y,domain,real_domain) in enumerate(train_dataloader):
                net.train()
                mini_batches = [x.cuda(gpu), y.cuda(gpu),domain.cuda(gpu),real_domain.cuda(gpu)]
                if args.amp:
                    step_vals = net.update(mini_batches,scaler)
                else:
                    step_vals = net.update(mini_batches)
                if (i+1) % 100 == 0 and gpu==0:
                    log_str = "Step %d " % (step+1)
                    for k, v in step_vals.items():
                        log_str = log_str + "%s: %.4f, " % (k, v)
                    print(log_str)
                    dict={'epoch':step,'model':net.network.module.state_dict(),'optimizer':net.optimizer.state_dict()}
                    torch.save(dict, "run2.pth")
            net.scheduler.step(step)
            if (step+1) % 1 == 0 and gpu==0:
                accuracy = helper.test(net, test_dataloader, gpu)
                if accuracy>best_acc:
                    best_acc=accuracy
                    dict={'epoch':step,'model':net.network.module.state_dict(),'optimizer':net.optimizer.state_dict()}
                    torch.save(dict, "results/kd_best5.pth")
                print("ite: %d, test accuracy: %.4f" % (step+1, accuracy))


if __name__=="__main__":
    args = helper.Args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    port_id = 10000 + np.random.randint(0, 1000)
    dist_url = 'tcp://127.0.0.1:' + str(port_id)
    distributed = True
    ngpus_per_node = torch.cuda.device_count()
    print('ngpus_per_node', ngpus_per_node)
    world_size=1
    rank=0
    world_size = ngpus_per_node * world_size
    print('multiprocessing_distributed')
    torch.multiprocessing.set_start_method('spawn')
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node,rank,world_size,dist_url,args))
