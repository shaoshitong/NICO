import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import torch.autograd as autograd
import copy
from pytorch_optimizer import AdaBound
import helper,einops
from timm.scheduler import CosineLRScheduler


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def KL(pred,target):
    return F.kl_div(torch.log_softmax(pred,1),target,reduction="batchmean")

class Featurizer(nn.Module):
    def __init__(self):
        super(Featurizer, self).__init__()
        self.network = torchvision.models.resnet18(pretrained=False)
        self.output_dim = self.network.fc.in_features
        del self.network.fc
        self.network.fc = Identity()

    def forward(self, x):
        return self.network(x)

    def get_output_dim(self):
        return self.output_dim


class Classifier(nn.Module):
    def __init__(self, in_features, out_features):
        super(Classifier, self).__init__()
        self.layer = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.layer(x)


class ERM(nn.Module):
    def __init__(self, num_classes, args):
        super(ERM, self).__init__()
        self.featurizer = Featurizer()
        features_in = self.featurizer.get_output_dim()
        self.classifier = Classifier(features_in, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                         momentum=args.momentum)

    def update(self, minibatches):
        all_x, all_y = minibatches
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)


class DenseNetM(nn.Module):
    def __init__(self, num_classes, args):
        super(DenseNetM, self).__init__()
        from othermodel.DenseNet import DenseNetV2
        self.network = DenseNetV2(num_classes=num_classes)
        if args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.network.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                             momentum=args.momentum)
        else:
            self.optimizer = AdaBound(self.network.parameters(), lr=args.lr * 0.01, final_lr=args.lr,
                                      weight_decay=args.weight_decay)
        self.scheduler = CosineLRScheduler(self.optimizer, t_initial=args.num_steps, lr_min=1e-4, cycle_mul=1.,
                                           warmup_lr_init=1e-4, warmup_t=10, t_in_epochs=True, cycle_limit=1)
        self.iter_nums = 0.
        self.queue=[torch.zeros(1,128) for _ in range(6)]
        self.queuelen=args.queuelen
        self.sample_num=args.sample_num
        self.kl_loss=nn.KLDivLoss(reduction="batchmean")
        self.temperature=args.temperature
    def checkqueue(self):
        for i,subqueue in enumerate(self.queue):
            if subqueue.shape[0]>self.queuelen:
                self.queue[i]=self.queue[i][-self.queuelen:]
    def enqueue(self,all_domains,domain):
        domain=domain.tolist()
        for i,do in enumerate(domain):
            self.queue[do]=torch.cat([self.queue[do],all_domains[i].detach().cpu().unsqueeze(0)],0)
        self.checkqueue()
    def dosample(self):
        sample_result=[]
        subsamlenum=int(self.sample_num//6)
        for i in range(6):
            subqueuelen=len(self.queue[i])
            if subqueuelen>subsamlenum:
                index=np.random.choice(subqueuelen,subsamlenum,replace=False)
            else:
                index=np.random.choice(subqueuelen,subsamlenum,replace=True)
            sample_sub_result=self.queue[i][index]
            sample_result.append(sample_sub_result)
        return torch.stack(sample_result,0)
    def update(self, minibatches, scaler: torch.cuda.amp.GradScaler = None):
        if scaler == None:
            all_x, all_y, domain ,real_domain = minibatches
            all_logits = self.predict(all_x) # all_domain: bs,128
            cls_loss = KL(all_logits, all_y)
            loss = cls_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return {'cls_loss': cls_loss.item()}
        else:
            all_x, all_y, domain ,real_domain = minibatches
            with torch.cuda.amp.autocast(enabled=True):
                all_logits = self.predict(all_x) # all_domain: bs,128
                cls_loss = KL(all_logits, all_y)
                loss = cls_loss
            self.optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            return {'cls_loss': cls_loss.item()}

    def predict(self, x):
        return self.network(x)

class PyramidNet(nn.Module):
    def __init__(self, num_classes, args,blocks=272,alpha=200,shakedrop=False):
        super(PyramidNet, self).__init__()
        from othermodel.pyramidnet import get_pyramidnet
        self.network = get_pyramidnet(blocks=blocks,alpha=alpha,num_classes=num_classes,shakedrop=shakedrop)
        if args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.network.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                             momentum=args.momentum)
        else:
            self.optimizer = AdaBound(self.network.parameters(), lr=args.lr * 0.01, final_lr=args.lr,
                                      weight_decay=args.weight_decay)
        self.scheduler = CosineLRScheduler(self.optimizer, t_initial=args.num_steps, lr_min=1e-4, cycle_mul=1.,
                                           warmup_lr_init=1e-4, warmup_t=10, t_in_epochs=True, cycle_limit=1)
        self.iter_nums = 0.
        self.queue=[torch.zeros(1,128) for _ in range(6)]
        self.queuelen=args.queuelen
        self.sample_num=args.sample_num
        self.kl_loss=nn.KLDivLoss(reduction="batchmean")
        self.temperature=args.temperature
    def checkqueue(self):
        for i,subqueue in enumerate(self.queue):
            if subqueue.shape[0]>self.queuelen:
                self.queue[i]=self.queue[i][-self.queuelen:]
    def enqueue(self,all_domains,domain):
        domain=domain.tolist()
        for i,do in enumerate(domain):
            self.queue[do]=torch.cat([self.queue[do],all_domains[i].detach().cpu().unsqueeze(0)],0)
        self.checkqueue()
    def dosample(self):
        sample_result=[]
        subsamlenum=int(self.sample_num//6)
        for i in range(6):
            subqueuelen=len(self.queue[i])
            if subqueuelen>subsamlenum:
                index=np.random.choice(subqueuelen,subsamlenum,replace=False)
            else:
                index=np.random.choice(subqueuelen,subsamlenum,replace=True)
            sample_sub_result=self.queue[i][index]
            sample_result.append(sample_sub_result)
        return torch.stack(sample_result,0)
    def update(self, minibatches, scaler: torch.cuda.amp.GradScaler = None):
        if scaler == None:
            all_x, all_y, domain ,real_domain = minibatches
            all_logits = self.predict(all_x) # all_domain: bs,128
            cls_loss = KL(all_logits, all_y)
            loss = cls_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return {'cls_loss': cls_loss.item()}
        else:
            all_x, all_y, domain ,real_domain = minibatches
            with torch.cuda.amp.autocast(enabled=True):
                all_logits = self.predict(all_x) # all_domain: bs,128
                cls_loss = KL(all_logits, all_y)
                loss = cls_loss
            self.optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            return {'cls_loss': cls_loss.item()}

    def predict(self, x):
        return self.network(x)



class PWDenseNetM(nn.Module):
    def __init__(self, num_classes, args):
        super(PWDenseNetM, self).__init__()
        from othermodel.DenseNet import DenseNetV2
        self.network = DenseNetV2(num_classes=num_classes)
        if args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.network.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                             momentum=args.momentum)
        else:
            self.optimizer = AdaBound(self.network.parameters(), lr=args.lr * 0.01, final_lr=args.lr,
                                      weight_decay=args.weight_decay)
        self.scheduler = CosineLRScheduler(self.optimizer, t_initial=args.num_steps, lr_min=1e-4, cycle_mul=1.,
                                           warmup_lr_init=1e-4, warmup_t=10, t_in_epochs=True, cycle_limit=1)
        self.iter_nums = 0.
        self.queue=[torch.zeros(1,128) for _ in range(6)]
        self.queuelen=args.queuelen
        self.sample_num=args.sample_num
        self.kl_loss=nn.KLDivLoss(reduction="batchmean")
        self.temperature=args.temperature
    def checkqueue(self):
        for i,subqueue in enumerate(self.queue):
            if subqueue.shape[0]>self.queuelen:
                self.queue[i]=self.queue[i][-self.queuelen:]
    def enqueue(self,all_domains,domain):
        domain=domain.tolist()
        for i,do in enumerate(domain):
            self.queue[do]=torch.cat([self.queue[do],all_domains[i].detach().cpu().unsqueeze(0)],0)
        self.checkqueue()
    def dosample(self):
        sample_result=[]
        subsamlenum=int(self.sample_num//6)
        for i in range(6):
            subqueuelen=len(self.queue[i])
            if subqueuelen>subsamlenum:
                index=np.random.choice(subqueuelen,subsamlenum,replace=False)
            else:
                index=np.random.choice(subqueuelen,subsamlenum,replace=True)
            sample_sub_result=self.queue[i][index]
            sample_result.append(sample_sub_result)
        return torch.stack(sample_result,0)
    def update(self, minibatches, scaler: torch.cuda.amp.GradScaler = None):
        if scaler == None:
            all_x, all_y, domain ,real_domain = minibatches
            all_logits = self.predict(all_x) # all_domain: bs,128
            cls_loss = KL(all_logits, all_y)
            loss = cls_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return {'cls_loss': cls_loss.item()}
        else:
            all_x, all_y, domain ,real_domain = minibatches
            with torch.cuda.amp.autocast(enabled=True):
                all_logits = self.predict(all_x) # all_domain: bs,128
                cls_loss = KL(all_logits, all_y)
                loss = cls_loss
            self.optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            return {'cls_loss': cls_loss.item()}

    def predict(self, x):
        return self.network(x)



class DenseNetFinetune(nn.Module):
    def __init__(self, num_classes, args):
        super(DenseNetFinetune, self).__init__()
        from othermodel.DenseNet import DenseNetV2
        self.network = DenseNetV2(num_classes=num_classes)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-4, weight_decay=0)
        self.iter_nums = 0.
        self.queue=[torch.zeros(1,128) for _ in range(6)]
        self.queuelen=args.queuelen
        self.sample_num=args.sample_num
        self.kl_loss=nn.KLDivLoss(reduction="batchmean")
        self.temperature=args.temperature
    def checkqueue(self):
        for i,subqueue in enumerate(self.queue):
            if subqueue.shape[0]>self.queuelen:
                self.queue[i]=self.queue[i][-self.queuelen:]
    def enqueue(self,all_domains,domain):
        domain=domain.tolist()
        for i,do in enumerate(domain):
            self.queue[do]=torch.cat([self.queue[do],all_domains[i].detach().cpu().unsqueeze(0)],0)
        self.checkqueue()
    def dosample(self):
        sample_result=[]
        subsamlenum=int(self.sample_num//6)
        for i in range(6):
            subqueuelen=len(self.queue[i])
            if subqueuelen>subsamlenum:
                index=np.random.choice(subqueuelen,subsamlenum,replace=False)
            else:
                index=np.random.choice(subqueuelen,subsamlenum,replace=True)
            sample_sub_result=self.queue[i][index]
            sample_result.append(sample_sub_result)
        return torch.stack(sample_result,0)
    def update(self, minibatches, scaler: torch.cuda.amp.GradScaler = None):
        all_x, all_y, domain ,real_domain = minibatches
        with torch.cuda.amp.autocast(enabled=True):
            all_logits = self.predict(all_x) # all_domain: bs,128
            cls_loss = KL(all_logits, all_y)
            loss = cls_loss
        self.optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(self.optimizer)
        scaler.update()
        return {'cls_loss': cls_loss.item(),'sum_acc':(all_logits.argmax(1)==all_y.argmax(1)).sum().item()}

    def predict(self, x):
        return self.network(x)



class Mixup(ERM):
    """
    Mixup of minibatches from different domains (https://github.com/facebookresearch/DomainBed/blob/25f173caa689f20828629b2e42f90193f203fdfa/domainbed/algorithms.py#L410)
    https://arxiv.org/pdf/2001.00677.pdf
    https://arxiv.org/pdf/1912.01805.pdf
    """

    def __init__(self, num_classes, args):
        super(Mixup, self).__init__(num_classes, args)
        self.args = args

    def update(self, minibatches, unlabeled=None):
        loss = 0

        for idx, minibatches_domain in enumerate(minibatches):  # randomize the minibatch
            rand_idx = torch.randperm(len(minibatches_domain[0]))
            minibatches[idx] = [minibatches_domain[0][rand_idx], minibatches_domain[1][rand_idx]]

        for (xi, yi), (xj, yj) in helper.random_pairs_of_minibatches(minibatches):
            lam = np.random.beta(self.args.mixup_alpha, self.args.mixup_alpha)
            xi, yi, xj, yj = xi.cuda(), yi.cuda(), xj.cuda(), yj.cuda()
            x = lam * xi + (1 - lam) * xj
            predictions = self.predict(x)

            loss += lam * F.cross_entropy(predictions, yi)
            loss += (1 - lam) * F.cross_entropy(predictions, yj)

        loss /= len(minibatches)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}
