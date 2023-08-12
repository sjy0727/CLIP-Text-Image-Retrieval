#!/usr/bin/env python
# -*- encoding : utf-8 -*-
"""
@Author :sunjunyi
@Time   :2023/8/6 14:00
"""
import os
import yaml
import json
import pandas as pd

from config import config
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoProcessor

# # 加载配置文件
# with open('config.yaml', 'r', encoding='utf-8') as f:
#     config = yaml.safe_load(f)


def build_image_caption_pair(is_train=False):
    root_dir = './mini-imagenet/'

    with open(os.path.join(root_dir, 'classes_name.json'), 'r') as f:
        mini_imagenet_label = json.load(f)

    csv_filepath = 'new_train.csv' if is_train else 'new_val.csv'

    data = pd.read_csv(os.path.join(root_dir, csv_filepath))

    res = []
    for idx, (_, row) in enumerate(data.iterrows()):
        img_path = os.path.join(root_dir, 'images', row['filename'])
        category = int(mini_imagenet_label[row['label']][0])
        label = mini_imagenet_label[row['label']][1].replace('_', ' ')
        res.append((idx, category, img_path, label))
    return res


class ImageCaptionDataset(Dataset):
    def __init__(self, is_train=False, return_loss=False):
        self.data = build_image_caption_pair(is_train=is_train)
        self.return_loss = return_loss
        self.processor = AutoProcessor.from_pretrained(config.finetune.checkpoint_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx, category, img_path, label = self.data[idx]
        img = Image.open(img_path)
        return idx, category, img, label

    def collate_fn(self, batch):
        ids, categories, images, labels = tuple(zip(*batch))  # batch接收的是dataset getitem方法返回值的列表
        output = self.processor(text=labels, images=images, return_tensors='pt', padding=True)
        if self.return_loss:
            output['return_loss'] = True
        return ids, categories, output