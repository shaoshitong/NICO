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

def main_worker(gpu, ngpus_per_node,rank,world_size,dist_url,args):
    rank = rank * ngpus_per_node + gpu
    dist.init_process_group(backend='nccl', init_method=dist_url,
                        world_size=world_size, rank=rank)
    net = model.DenseNetFinetune(args.num_classes, args)
    net.cuda(gpu)
    torch.backends.cudnn.benchmark=True
    torch.backends.cudnn.fastest=True
    net.network = torch.nn.parallel.DistributedDataParallel(net.network, device_ids=[gpu])
    args.batch_size = int(args.batch_size / ngpus_per_node)
    _,train_dataloader = helper.get_our_dataloader(args,'/home/sst/dataset/nico/nico/train' ,'train')
    train_dataloader.dataset.val=True
    print("begin training......")
    dict=torch.load("./results/best.pth")
    net.network.module.load_state_dict(dict['model'])
    for step in range(1):
        data_number = 0.
        acc_number = 0.
        if gpu==0:
            print(f"lr:{net.optimizer.param_groups[0]['lr']}")
        for i,(x,y,domain,real_domain) in enumerate(train_dataloader):
            net.train()
            mini_batches = [x.cuda(gpu), y.cuda(gpu),domain.cuda(gpu),real_domain.cuda(gpu)]
            if args.amp:
                step_vals = net.update(mini_batches,scaler)
            else:
                step_vals = net.update(mini_batches)
            data_number+=x.shape[0]
            acc_number+=step_vals['sum_acc']
            if (i+1) % 100 == 0 and gpu==0:
                log_str = "Step %d " % (step+1)
                for k, v in step_vals.items():
                    log_str = log_str + "%s: %.4f, " % (k, v)
                print(log_str)
                dict={'epoch':step,'model':net.network.module.state_dict(),'optimizer':net.optimizer.state_dict()}
                torch.save(dict, "./results/finetune.pth")
        print("real acc",round(acc_number/data_number,3))
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
