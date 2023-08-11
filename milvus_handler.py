#!/usr/bin/env python
# -*- encoding : utf-8 -*-
"""
@Author :sunjunyi
@Time   :2023/8/11 21:06
"""
import yaml
from pymilvus import MilvusClient, connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# 加载配置文件
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)


class MilvusHandler:
    def __init__(self,
                 host=config['milvus']['host'],
                 port=config['milvus']['port']
                 ):
        # 连接向量数据库并加载到内存中
        connections.connect(host=host, port=port)
        self.collection = Collection(config['milvus']['collection_name'])
        self.collection.load()

    def search(self, embeds, topk):
        res = self.collection.search(  # TODO: 把milvus部分代码抽离成单个类
            data=embeds,
            anns_field='embedding',
            param=config['milvus']['search_params'],
            limit=topk,
            output_fields=['category']
        )

        ids = [list(hits.ids) for hits in res]
        distances = [list(hits.distances) for hits in res]
        categories = [[hit.entity.get('category') for hit in hits] for hits in res]
        return ids, distances, categories

    def insert(self, data):  # 可以传list(dict) or list(list)
        self.collection.insert(data=data)
        self.collection.load()
        self.collection.flush()

    @staticmethod
    def list_collections():
        utility.list_collections()

    @staticmethod
    def drop_collection(collection_name):
        utility.drop_collection(collection_name)
