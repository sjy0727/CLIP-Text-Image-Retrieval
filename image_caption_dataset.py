#!/usr/bin/env python
# -*- encoding : utf-8 -*-
"""
@Author :sunjunyi
@Time   :2023/8/6 14:00
"""
import os
import json
import pandas as pd

from config import config
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoProcessor


def get_labels_and_cates(dataset, is_train):
    build_dataset_fn = {
        'mini imagenet': build_mini_imagenet_dataset,
        'coco': build_coco_dataset,
        'flickr30k': build_flickr30k
    }
    array = build_dataset_fn[dataset](is_train=is_train)
    array = sorted(array, key=lambda x: int(x[1]))
    labels = [item[3] for item in array]
    cates = [item[1] for item in array]
    return labels, cates


def build_mini_imagenet_dataset(is_train=False):
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


def build_coco_dataset(is_train=False):
    # 设置数据集路径
    dataDir = './MSCOCO'  # 将路径替换为您的数据集路径
    dataType = 'train2014' if is_train else 'val2014'  # 或者 'val2014'，取决于您想要加载的数据集部分
    annFile = os.path.join(dataDir, f'annotations/captions_{dataType}.json')

    # 加载注释文件
    with open(annFile, 'r') as f:
        annotations = json.load(f)['annotations']

    # 构建图像ID到描述的映射
    imgid_to_captions = {}
    for ann in annotations:
        img_id = ann['image_id']
        caption = ann['caption']
        if img_id in imgid_to_captions:
            imgid_to_captions[img_id].append(caption)  # 对应多条描述
        else:
            imgid_to_captions[img_id] = [caption]

    # 加载图像和对应的描述
    res = []
    img_folder = os.path.join(dataDir, dataType)
    for idx, (img_id, captions) in enumerate(imgid_to_captions.items()):
        img_info = {
            'id': img_id,
            'file_name': f'{img_id:012d}.jpg'  # COCO图像文件名的格式
        }
        img_path = os.path.join(img_folder, img_info['file_name'])

        for caption in captions:
            category = img_id  # img_id作为类别数字
            res.append((idx, category, img_path, caption))
    return sorted(res, key=lambda x: int(x[1]))

    # images_and_captions 中包含了每张图像及其对应的多个描述
    # (idx, category, img_path, label)


def build_flickr30k(is_train=False):
    dataDir = './Flickr30k'
    annFile = os.path.join(dataDir, 'results_20130124.token')
    img_folder = os.path.join(dataDir, 'flickr30k-images')
    annotations = pd.read_table(annFile, sep='\t', header=None, names=['image', 'caption'])
    captions2images = {caption: image.split('#')[0] for caption, image in
                       zip(annotations['caption'], annotations['image'])}

    res = []
    for idx, (image, caption) in enumerate(captions2images.items()):
        img_path = os.path.join(img_folder, image)
        category = int(image.split('.')[0])
        res.append((idx, category, img_path, caption))
    return res


class ImageCaptionDataset(Dataset):
    def __init__(self, is_train=False, return_loss=False, dataset='mini imagenet'):
        build_dataset_fn = {
            'mini imagenet': build_mini_imagenet_dataset,
            'coco': build_coco_dataset,
            'flickr30k': build_flickr30k
        }
        self.data = build_dataset_fn[dataset](is_train=is_train)
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
