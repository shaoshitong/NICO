import os, copy

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


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


class NoisyStudent:
    def __init__(
            self,
            gpu,
            total_tta=80,
            label2id_path: str = "/rooot/autodl/dg_label_id_mapping.json",
            test_image_path: str = "/home/Bigdata/NICO/nico/test/",
            batch_size: int = 64,
            track_mode="track1",
            img_size=224,
            parallel=False,
    ):
        self.result = {}
        label2id_path: str = label2id_path
        test_image_path: str = test_image_path
        self.test_loader_predict, _ = get_test_loader(
            batch_size=batch_size,
            transforms=None,
            label2id_path=label2id_path,
            test_image_path=test_image_path,
            img_size=img_size,
            track_mode=track_mode,
        )
        self.test_loader_student, self.label2id = get_test_loader(
            batch_size=batch_size,
            transforms="train",
            img_size=img_size,
            label2id_path=label2id_path,
            test_image_path=test_image_path,
            track_mode=track_mode,
        )
        self.gpu = gpu
        self.total_tta = total_tta
        self.parallel = parallel
        self.model = pyramidnet272(num_classes=60, num_models=-1).cuda(self.gpu)

    def save_result(self, path="prediction.json"):
        result = {}
        for name, pre in list(self.result.items()):
            _, y = torch.max(pre, dim=1)
            result[name] = y.item()
        if self.parallel:
            path1,path2=path.split(".")
            write_result(result, path=f"{path1}_gpu_" + str(self.gpu) + ".json")
        else:
            write_result(result, path=path)
        return result

    def predict(self):
        with torch.no_grad():
            print("teacher are giving his predictions!")
            self.model.eval()
            for x, names in tqdm(self.test_loader_predict):
                x = x.cuda(self.gpu)
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

    @torch.no_grad()
    def TTA(self, aug_weight=0.5):
        self.predict()
        print("now we are doing TTA")
        for epoch in range(1, self.total_tta + 1):
            self.model.eval()
            for x, names in tqdm(self.test_loader_student):
                x = x.cuda(self.gpu)
                x = self.model(x)
                for i, name in enumerate(list(names)):
                    self.result[name] += x[i, :].unsqueeze(0) * aug_weight
            print(f"Epoch {epoch} finished")
        print("TTA finished")


def main_worker(
        gpu,
        ngpus_per_node,
        batch_size,
        total_tta,
        dist_url,
        world_size,
        label2id_path,
        test_pth_path,
        test_image_path,
        img_size,
        track_mode,
        json_save_path,
):
    print("Use GPU: {} for training".format(gpu))
    rank = 0  # 单机
    dist_backend = "nccl"
    rank = rank * ngpus_per_node + gpu
    print("world_size:", world_size)
    dist.init_process_group(
        backend=dist_backend, init_method=dist_url, world_size=world_size, rank=rank)
    torch.cuda.set_device(gpu)
    batch_size = int(batch_size / ngpus_per_node)
    print("sub batch size is", batch_size)
    x = NoisyStudent(
        gpu=gpu,
        batch_size=batch_size,
        label2id_path=label2id_path,
        test_image_path=test_image_path,
        img_size=img_size,
        total_tta=total_tta,
        track_mode=track_mode,
        parallel=True,
    )

    if not os.path.exists(test_pth_path):
        raise FileNotFoundError("test pth path can not be found")
    dict = torch.load(test_pth_path)
    x.model.load_state_dict(dict['model'])
    x.model = DDP(x.model, device_ids=[gpu], output_device=0)
    test_predict_sampler = torch.utils.data.distributed.DistributedSampler(x.test_loader_predict.dataset)
    x.test_loader_predict = torch.utils.data.DataLoader(
        x.test_loader_predict.dataset,
        batch_size=batch_size,
        sampler=test_predict_sampler,
        num_workers=6 if batch_size % 6 == 0 else 4,
        pin_memory=True,
    )
    test_student_sampler = torch.utils.data.distributed.DistributedSampler(x.test_loader_student.dataset)
    x.test_loader_student = torch.utils.data.DataLoader(
        x.test_loader_student.dataset,
        batch_size=batch_size,
        sampler=test_student_sampler,
        num_workers=6 if batch_size % 6 == 0 else 4,
        pin_memory=True,
    )
    x.TTA()
    x.save_result(path=json_save_path)


def sum(aps, bps):
    keys = aps.keys()
    result = copy.deepcopy(aps)
    for key in keys:
        result[key] = (aps[key] + bps[key]) / 2
    return result


if __name__ == "__main__":
    import argparse

    # 86.46
    paser = argparse.ArgumentParser()
    paser.add_argument("--batch_size", default=128, type=int)
    paser.add_argument("--total_tta", default=0, type=int)
    paser.add_argument("--parallel", default=False, action="store_true")
    paser.add_argument("--img_size", default=32, type=int)
    paser.add_argument("--cuda_devices", default="0,1", type=str)
    paser.add_argument("--test_pth_path", default='original_1.pth', type=str)
    paser.add_argument("--track_mode", default="track2", type=str)
    paser.add_argument("--json_save_path",default='prediction_2.json',type=str)
    paser.add_argument(
        "--label2id_path", default="/home/Bigdata/NICO2/ood_label_id_mapping.json", type=str
    )
    paser.add_argument("--test_image_path", default="/home/Bigdata/NICO2/nico/test/", type=str)
    args = paser.parse_args()

    print(args)
    batch_size = args.batch_size
    total_tta = args.total_tta
    parallel = args.parallel
    json_save_path = args.json_save_path
    if parallel:
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
                total_tta,
                dist_url,
                world_size,
                args.label2id_path,
                args.test_pth_path,
                args.test_image_path,
                args.img_size,
                args.track_mode,
                json_save_path,
            ),
        )
        import json
        last_result = {}
        path1,path2=json_save_path.split(".")
        for gpu in args.cuda_devices.split(','):
            with open(f"{path1}_gpu_{gpu}.json", "r") as f:
                result = json.load(f)
                last_result.update(result)
        with open(json_save_path, "w") as f:
            json.dump(last_result, f)
        for gpu in args.cuda_devices.split(','):
            os.system(f'rm {path1}_gpu_{gpu}.json')
    else:
        x = NoisyStudent(
            gpu=0,
            batch_size=batch_size,
            label2id_path=args.label2id_path,
            test_image_path=args.test_image_path,
            img_size=args.img_size,
            total_tta=args.total_tta,
            track_mode=args.track_mode,
            parallel=False,
        )
        if not os.path.exists(args.test_pth_path):
            raise FileNotFoundError("test pth path can not be found")
        dict = torch.load(args.test_pth_path)
        dict1 = torch.load("original_1.pth")['model']
        dict2 = torch.load("original_2.pth")['model']
        dict3 = sum(dict1, dict2)
        x.model.load_state_dict(dict3)
        x.TTA()
        x.save_result(path=json_save_path)
