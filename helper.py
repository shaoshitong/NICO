import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
from os.path import join, dirname
import numpy as np
import random
import hashlib
from utils.policy import CIFAR10Policy
from sklearn.model_selection import StratifiedShuffleSplit
import os
from torchvision.datasets import ImageFolder


def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2 ** 31)


def _hparam(name, default_val, random_val_fn, random_seed=0):
    """Define a hyperparameter. random_val_fn takes a RandomState and
    returns a random hyperparameter value."""
    random_state = np.random.RandomState(seed_hash(random_seed, name))
    return (default_val, random_val_fn(random_state))


class Args():
    def __init__(self):
        self.gpu_id = 0
        self.batch_size = 42
        self.num_classes = 60
        self.num_steps = 200
        self.lr = 0.09*48/42
        self.resume=True
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.optimizer = "sgd"
        self.min_scale = 0.8
        self.max_scale = 1.0
        self.random_horiz_flip = 0.5
        self.jitter = 0.4        
        self.image_size = 224
        self.queuelen=1
        self.sample_num=1
        self.amp=True
        self.num_domains = 6
        self.mixup_alpha = 0.2
        self.temperature=5
        self.KD=True



def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def _dataset_info(txt_file):
    with open(txt_file, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []
    for row in images_list:
        row = row.split(' ')
        file_names.append(row[0])
        labels.append(int(row[1]))

    return file_names, labels



### for ERM w/o domain information

class MyDataset(data.Dataset):
    def __init__(self, names, labels, img_transformer=None):
        self.names = names
        self.labels = labels

        self.N = len(self.names)
        self._image_transformer = img_transformer
    
    def get_image(self, index):
        img = Image.open(self.names[index]).convert('RGB')
        return self._image_transformer(img)
        
    def __getitem__(self, index):
        img = self.get_image(index)
        return img, int(self.labels[index])

    def __len__(self):
        return len(self.names)

class DGDataset(data.Dataset):
    def __init__(self,result,img_transformer=None,val_transformer=None):
        self.result=result
        self._image_transformer=img_transformer
        self.val_transformer=val_transformer
        self.train=isinstance(self.result[0],list)
        self.nums=10
        self.beta=0.3
        self.val=False
        self.eps=1.
    def __len__(self):
        return len(self.result)

    def YOCO(self,img):
        q = self._image_transformer(img)
        k = self._image_transformer(img)
        c,h,w=q.size()
        if np.random.random() < 0.5:
            q = torch.cat([q[:,:,0:int(w/2)],k[:,:,int(w/2):w]],dim=2)
        else:
            q = torch.cat([q[:,0:int(h/2),:],k[:,int(h/2):h,:]],dim=1)
        return q
    def __getitem__(self,index):
        if self.train:
            img,target,domain=self.result[index]
            label=torch.zeros(60)
            label[target]=1
            img =Image.open(img).convert('RGB')
            img=self.YOCO(img) if self.eps>random.random() else self.val_transformer(img)
            m_domain = torch.zeros(6)
            m_domain[domain] = 1
            if index > 0 and index % self.nums ==0 and self.val==False:
                mixup_idx = random.randint(0,len(self.result)-1)
                mix_img, mix_target, mix_domain = self.result[mixup_idx]
                mix_label=torch.zeros(60)
                mix_label[mix_target]=1
                mix_img=Image.open(mix_img).convert('RGB')
                mix_img=self.YOCO(mix_img) if self.eps>random.random() else self.val_transformer(mix_img)
                lam=np.random.beta(self.beta,self.beta)
                img_h,img_w=img.shape[1:]
                cx=np.random.uniform(0,img_w)
                cy=np.random.uniform(0,img_h)
                w=img_w*np.sqrt(1-lam)
                h=img_h*np.sqrt(1-lam)
                x0 = int(np.round(max(cx - w / 2, 0)))
                x1 = int(np.round(min(cx + w / 2, img_w)))
                y0 = int(np.round(max(cy - h / 2, 0)))
                y1 = int(np.round(min(cy + h / 2, img_h)))
                area=(x1-x0)*(y1-y0)
                tarea=img_w*img_h
                img[:, y0:y1, x0:x1] = mix_img[:, y0:y1, x0:x1]
                label = label*(1-(area/tarea)**2)+(area/tarea)*mix_label
                m_mix_domain=torch.zeros(6)
                m_mix_domain[mix_domain]=1
                return img,label,domain,m_domain*(1-(area/tarea)**2)+(area/tarea)*m_mix_domain
            else:
                return img,label,domain,m_domain
        else:
            img_path=self.result[index]
            img =Image.open(img_path).convert('RGB')
            img=self._image_transformer(img)
            return img,img_path

class EnsembleDataset(data.Dataset):
    def __init__(self, result):
        self.result=result
        self.val_transformer=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.val_list=[transforms.ColorJitter(brightness=random.random()/2,contrast=random.random()/2,hue=random.random()/2) for i in range(10)]
    def __len__(self):
        return len(self.result)
    def __getitem__(self, item):
        img_path=self.result[item]
        img=Image.open(img_path).convert('RGB')
        img_list=[self.val_transformer(self.val_list[i](img)) for i in range(10)]
        img_list=torch.stack(img_list)
        return img_list,img_path
class InfDataLoader():
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.data_iter = iter(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            data = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            data = next(self.data_iter)
        return data
        
    def __len__(self):
        return len(self.dataloader)


def get_train_transformer(args):
    img_tr = [transforms.RandomResizedCrop((int(args.image_size), int(args.image_size)), (args.min_scale, args.max_scale))]
    if args.random_horiz_flip > 0.0:
        img_tr.append(transforms.RandomHorizontalFlip(args.random_horiz_flip))
    if args.jitter > 0.0:
        img_tr.append(transforms.ColorJitter(brightness=args.jitter, contrast=args.jitter, saturation=args.jitter, hue=min(0.5, args.jitter)))
    img_tr = img_tr + [CIFAR10Policy(),transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    return transforms.Compose(img_tr)

def get_val_transformer(args):
    img_tr = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return transforms.Compose(img_tr)

def get_ERM_dataloader(args, phase,num_worker=8):
    assert phase in ["train", "test"]
    names, labels = _dataset_info('data/ERM_' + phase + '.txt')

    if phase == "train":
        img_tr = get_train_transformer(args)
    else:
        img_tr = get_val_transformer(args)
    mydataset = MyDataset(names, labels, img_tr)
    do_shuffle = True if phase == "train" else False
    if phase == "train":
        loader = InfDataLoader(mydataset, batch_size=args.batch_size, shuffle=do_shuffle, num_workers=num_worker)
    else:
        loader = data.DataLoader(mydataset, batch_size=args.batch_size, shuffle=do_shuffle, num_workers=4)
    return loader

def get_our_dataloader(args,path,phase,num_worker=8):
    assert phase in ["train", "test"]
    from utils.generate_txt_label import generate_test,generate_train
    if phase=='train':
        results=generate_train(path)
        img_tr=get_train_transformer(args)
        img_vr=get_val_transformer(args)
        dataset=DGDataset(results,img_tr,img_vr)
        labels = [int(dataset.result[i][1]) for i in range(len(dataset.result))]
        few_ratio=0.9
        ss = StratifiedShuffleSplit(n_splits=1, test_size=1 - few_ratio, random_state=0)
        train_indices, valid_indices = list(ss.split(np.array(labels)[:, np.newaxis], labels))[0]
        trainset = torch.utils.data.Subset(dataset, train_indices)
        testset = torch.utils.data.Subset(dataset, valid_indices)
        sampler=torch.utils.data.distributed.DistributedSampler(trainset)
    else:
        results=generate_test(path)
        img_tr=get_val_transformer(args)
        dataset=DGDataset(results,img_tr)
    do_shuffle = True if phase == "train" else False
    if phase == "train":
        loader1 = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=(sampler is None), num_workers=num_worker,pin_memory=True,sampler=sampler)
        loader2 = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4,pin_memory=True)
        return loader1,loader2
    else:
        loader = data.DataLoader(dataset, batch_size=2, shuffle=do_shuffle, num_workers=4,pin_memory=True)
        return loader

def test(network, dataloader, gpu):
    network.eval()
    corrects = 0
    with torch.no_grad():
        for images, labels, _ ,_ in dataloader:
            images, labels = images.cuda(gpu), labels.cuda(gpu)
            output = network.predict(images)
            _, predictions = output.max(dim=1)
            corrects += torch.sum(predictions == labels.argmax(1))
    accuracy = float(corrects) / len(dataloader.dataset)
    network.train()
    return accuracy




### for mixup w/ domain information

class MyDataset_DG(data.Dataset):
    def __init__(self, names, n_domains, labels, img_transformer=None):
        self.names = names
        self.labels = labels
        self.n_domains = n_domains

        self.N = len(self.names)
        self._image_transformer = img_transformer
        self.names_domains = [self.names[i*self.N//n_domains:(i+1)*self.N//n_domains] for i in range(n_domains)]
        self.labels_domain = [self.labels[i*self.N//n_domains:(i+1)*self.N//n_domains] for i in range(n_domains)]

    def get_image_domain(self, index, domain_index):
        img = Image.open(self.names_domains[domain_index][index]).convert('RGB')
        return self._image_transformer(img), self.labels_domain[domain_index][index]

    def __getitem__(self, index):
        input_set = []
        for i in range(self.n_domains):
            input_set.append(list(self.get_image_domain(index, i)))
        return input_set

    def __len__(self):
        return int(self.N / self.n_domains)



def get_val_dataloader(path,args):
    from utils.generate_txt_label import generate_test,generate_train
    results=generate_test(path)
    dataset=DGDataset(results,img_transformer=get_train_transformer(args))
    dataloaer=data.DataLoader(dataset,batch_size=128, shuffle=False, num_workers=4,pin_memory=True)
    return dataloaer

def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs
