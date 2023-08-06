#!/usr/bin/env python
# -*- encoding : utf-8 -*-
"""
@Author :sunjunyi
@Time   :2023/8/5 18:46
"""
import os
import json
import yaml

import pandas as pd
import torch

from image_caption_dataset import ImageCaptionDataset
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from pymilvus import MilvusClient, connections, FieldSchema, CollectionSchema, DataType, Collection, utility, db
from transformers import AutoModel, AutoProcessor, CLIPVisionModelWithProjection, CLIPTextModelWithProjection, CLIPModel

# 加载配置文件
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)


def create_milvus_collection(collection_name, dim):
    connections.connect(host=config['milvus']['host'], port=config['milvus']['port'])

    if utility.has_collection(collection_name):  # 如果数据库存在则删除
        utility.drop_collection(collection_name)

    fields = [
        FieldSchema(name='id', dtype=DataType.INT64, descrition='ids', is_primary=True, auto_id=False),
        FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, descrition='embedding vectors', dim=dim),
        FieldSchema(name='category', dtype=DataType.INT64, descrition='category'),
    ]

    schema = CollectionSchema(fields=fields, description='mini imagenet text image search')
    collection = Collection(name=collection_name, schema=schema)

    index_params = config['milvus']['index_params']

    # 根据字段建立索引
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection


# def build_image_caption_pair(is_train=False):
#     root_dir = './mini-imagenet/'
#
#     with open(os.path.join(root_dir, 'classes_name.json'), 'r') as f:
#         mini_imagenet_label = json.load(f)
#
#     csv_filepath = 'new_train.csv' if is_train else 'new_val.csv'
#
#     data = pd.read_csv(os.path.join(root_dir, csv_filepath))
#
#     res = []
#     for idx, (_, row) in enumerate(data.iterrows()):
#         img_path = os.path.join(root_dir, 'images', row['filename'])
#         category = int(mini_imagenet_label[row['label']][0])
#         label = mini_imagenet_label[row['label']][1].replace('_', ' ')
#         res.append((idx, category, img_path, label))
#     return res
#
#
# class ImageCaptionDataset(Dataset):
#     """
#     输入是图像文件地址列表，标签文字列表，输出是text embeds和image embeds
#     """
#
#     def __init__(self, is_train=False, return_loss=False):
#         self.data = build_image_caption_pair(is_train=is_train)
#         self.return_loss = return_loss
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         idx, category, img_path, label = self.data[idx]
#         img = Image.open(img_path)
#         return idx, category, img, label
#
#     def collate_fn(self, batch):
#         processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16")
#         ids, categories, images, labels = tuple(zip(*batch))  # batch接收的是dataset getitem方法返回值的列表
#         output = processor(text=labels, images=images, return_tensors='pt', padding=True)
#         if self.return_loss:
#             output['return_loss'] = True
#         return ids, categories, output


if __name__ == '__main__':
    # 权重路径
    checkpoint_dir = config['model']['checkpoint_dir']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ',device)
    # 模型加载权重
    # image_encoder = CLIPVisionModelWithProjection.from_pretrained(checkpoint_dir).to(device)
    # text_encoder = CLIPTextModelWithProjection.from_pretrained(checkpoint_dir).to(device)
    model = CLIPModel.from_pretrained(checkpoint_dir).to(device)
    # 预处理器
    processor = AutoProcessor.from_pretrained(checkpoint_dir)

    root_dir = './mini-imagenet/'
    # dataset = image_title_data_iter(root_dir, checkpoint_dir)
    dataset = ImageCaptionDataset(is_train=False, return_loss=False)

    dataloader = DataLoader(
        dataset,
        batch_size=128,
        num_workers=0,  # iter-style dataset 需要设置为0
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        collate_fn=dataset.collate_fn
    )

    # connections.connect(host='127.0.0.1', port='19530')
    create_milvus_collection(config['milvus']['collection_name'], config['milvus']['vector_dim'])
    collection = Collection(config['milvus']['collection_name'])
    print(utility.list_collections())

    # 调用显卡inference
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader)):
            ids, categories, inputs = batch
            inputs = inputs.to(device)

            output = model(**inputs)
            output.image_embeds /= output.image_embeds.norm(dim=-1, keepdim=True)
            image_embeds = output.image_embeds.squeeze().cpu().numpy()  # 注意要squeeze() 不然无法插入

            output.text_embeds /= output.text_embeds.norm(dim=-1, keepdim=True)
            text_embeds = output.text_embeds.squeeze().cpu().numpy()

            insert_datas = [ids, (image_embeds + text_embeds) / 2, categories]  # 可以传list(dict) or list(list)
            mr = collection.insert(data=insert_datas)

    collection.load()  # 加载向量数据到内存中
    collection.flush()
    print(collection.num_entities)
    print(utility.list_collections())
    # utility.drop_collection('test')
