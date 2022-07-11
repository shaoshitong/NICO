import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .dataUtils import *


class DGDataSet(Dataset):
    def __init__(
        self,
        mode="train",
        valid_category=None,
        label2id_path="/home/sst/dataset/nico/dg_label_id_mapping.json",
        test_image_path=None,
        train_image_path="/home/sst/dataset/nico/nico/train/",
        transform_type=None,
    ):
        """
        :param mode:  train? valid? test?
        :param valid_category: if train or valid, you must pass this parameter
        :param label2id_path:
        :param test_image_path:
        :param train_image_path:  must end by '/'
        """
        self.mode = mode
        self.transform_type = transform_type
        self.label2id = get_label2id(label2id_path)
        self.id2label = reverse_dic(self.label2id)
        self.train_image_path = train_image_path
        self.test_image_path = test_image_path
        self.nums = 10
        self.beta = 0.3
        self.transform = transforms.Compose(
            [
                # add transforms here
                transforms.RandomResizedCrop((224, 224), scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                # transforms.RandomRotation(15),
                # transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # if train or valid, synthesize a dic contain num_images * dic, each subdic contain:
        # path、category_id、context_category

    def synthesize_images_track1(self):
        self.images = {}
        for context_category, context_value in list(self.total_dic.items()):
            for category_name, image_list in list(context_value.items()):
                for img in image_list:
                    now_dic = {}
                    now_dic["path"] = (
                        self.train_image_path + context_category + "/" + category_name + "/" + img
                    )
                    now_dic["category_id"] = self.label2id[category_name]
                    now_dic["context_category"] = context_category
                    self.images[len(self.images)] = now_dic

    def synthesize_images_track2(self):
        self.images = {}
        for category_name, image_list in list(self.total_dic.items()):
            for img in image_list:
                now_dic = {}
                now_dic["path"] = self.train_image_path + category_name + "/" + img
                now_dic["category_id"] = self.label2id[category_name]
                now_dic["context_category"] = None
                self.images[len(self.images)] = now_dic

    def __len__(self):
        return len(self.images)

    def cutmix_and_yoco(self, img, label, index):
        if index > 0 and index % self.nums == 0:
            mixup_idx = random.randint(0, len(self.images) - 1)
            mixup_dic = self.images[mixup_idx]
            mixup_img = Image.open(mixup_dic["path"]).convert("RGB")
            mixup_img = self.YOCO(mixup_img) if 0.5 > random.random() else self.transform(mixup_img)
            mixup_label = torch.zeros(60)
            mixup_label[mixup_dic["category_id"]] = 1
            lam = np.random.beta(self.beta, self.beta)
            img_h, img_w = img.shape[1:]
            cx = np.random.uniform(0, img_w)
            cy = np.random.uniform(0, img_h)
            w = img_w * np.sqrt(1 - lam)
            h = img_h * np.sqrt(1 - lam)
            x0 = int(np.round(max(cx - w / 2, 0)))
            x1 = int(np.round(min(cx + w / 2, img_w)))
            y0 = int(np.round(max(cy - h / 2, 0)))
            y1 = int(np.round(min(cy + h / 2, img_h)))
            area = (x1 - x0) * (y1 - y0)
            tarea = img_w * img_h
            img[:, y0:y1, x0:x1] = mixup_img[:, y0:y1, x0:x1]
            label = label * (1 - (area / tarea)) + (area / tarea) * mixup_label
        return img, label

    def YOCO(self, img):
        q = self.transform(img)
        k = self.transform(img)
        c, h, w = q.size()
        if np.random.random() < 0.5:
            q = torch.cat([q[:, :, 0 : int(w / 2)], k[:, :, int(w / 2) : w]], dim=2)
        else:
            q = torch.cat([q[:, 0 : int(h / 2), :], k[:, int(h / 2) : h, :]], dim=1)
        return q

    def __getitem__(self, item):
        if self.mode == "test":
            img = Image.open(self.test_image_path + self.images[item]).convert("RGB")
            if self.transform_type == "test" or self.transform_type is None:
                img = self.test_transform(img)
            else:
                img = self.transform(img)
            return img, self.images[item]

        if self.mode == "train" or self.mode == "valid":
            img_dic = self.images[item]
            img = Image.open(img_dic["path"]).convert("RGB")
            img = self.YOCO(img) if 0.5 > random.random() else self.transform(img)
            y = torch.zeros(60)
            y[img_dic["category_id"]] = 1
            img, y = self.cutmix_and_yoco(img, y, item)
            return img, y

    def get_id2label(self):
        return self.id2label


class Track1DataSet(DGDataSet):
    def __init__(
        self,
        mode="train",
        valid_category=None,
        label2id_path="/home/sst/dataset/nico/dg_label_id_mapping.json",
        test_image_path=None,
        train_image_path="/home/sst/dataset/nico/nico/train/",
        transform_type=None,
    ):
        super(Track1DataSet, self).__init__(
            mode, valid_category, label2id_path, test_image_path, train_image_path, transform_type
        )
        if mode == "test":
            self.images = get_test_set_images(test_image_path)
        if mode == "train":
            self.total_dic = get_train_set_dic_track1(train_image_path)
            if valid_category is not None:
                del self.total_dic[valid_category]
            self.synthesize_images_track1()
        if mode == "valid":
            self.total_dic = get_train_set_dic_track1(train_image_path)
            self.total_dic = dict([(valid_category, self.total_dic[valid_category])])
            self.synthesize_images_track1()


class Track2DataSet(DGDataSet):
    def __init__(
        self,
        mode="train",
        valid_category=None,
        label2id_path="/home/sst/dataset/nico/dg_label_id_mapping.json",
        test_image_path=None,
        train_image_path="/home/sst/dataset/nico/nico/train/",
        transform_type=None,
    ):
        super(Track2DataSet, self).__init__(
            mode, valid_category, label2id_path, test_image_path, train_image_path, transform_type
        )
        if mode == "test":
            self.images = get_test_set_images(test_image_path)
        if mode == "train":
            self.total_dic = get_train_set_dic_track2(train_image_path)
            self.synthesize_images_track2()


def get_loader(
    train_image_path,
    valid_image_path,
    label2id_path,
    batch_size=32,
    valid_category="autumn",
    num_workers=8,
    track_mode="track1",
):
    """
    if you are familiar with me, you will know this function aims to get train loader and valid loader
    :return:
    """
    context_category_list = ["autumn", "dim", "grass", "outdoor", "rock", "water"]
    track_list = ["track1", "track2"]
    assert track_mode in track_list, "track_mode should be one of track1 and track2"
    if valid_category == "rand":
        valid_category = context_category_list[random.randint(0, len(context_category_list) - 1)]
    if track_mode == "track2":
        valid_category = None
        print("track2 not supports valid mode")
    print(f"we choose {valid_category} as valid, others as train")
    print("-" * 100)
    MyDataSet = Track1DataSet if track_mode == "track1" else Track2DataSet
    train_set = MyDataSet(
        mode="train",
        train_image_path=train_image_path,
        label2id_path=label2id_path,
        valid_category=valid_category,
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    if valid_category is None:
        return train_loader
    valid_set = MyDataSet(
        mode="valid",
        valid_category=valid_category,
        train_image_path=valid_image_path,
        label2id_path=label2id_path,
    )
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)

    print("managed to get loader!!!!!")
    print("-" * 100)
    return train_loader, valid_loader


def get_test_loader(
    batch_size=32,
    test_image_path="./public_dg_0416/public_test_flat/",
    label2id_path="./dg_label_id_mapping.json",
    transforms=None,
    track_mode="track1",
):
    track_list = ["track1", "track2"]
    assert track_mode in track_list, "track_mode should be one of track1 and track2"
    MyDataSet = Track1DataSet if track_mode == "track1" else Track2DataSet

    test_set = MyDataSet(
        mode="test",
        test_image_path=test_image_path,
        label2id_path=label2id_path,
        transform_type=transforms,
    )
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    return loader, test_set.get_id2label()
