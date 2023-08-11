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
                 port=config['milvus']['port'],
                 collection_name=config['milvus']['collection_name']
                 ):
        # 连接向量数据库并加载到内存中
        self.collection_name = collection_name
        connections.connect(host=host, port=port)
        # 连接对应的collection
        self._connect_collection(self.collection_name)

    def _connect_collection(self, collection_name):
        if utility.has_collection(collection_name):  # 如果数据库存在则删除
            self.collection = Collection()
            self.collection.load()
        else:
            print('milvus中没有' + collection_name + '对应的collection')

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
    def create_collection(collection_name, dim):
        if utility.has_collection(collection_name):  # 如果数据库存在则删除
            utility.drop_collection(collection_name)

        fields = [
            FieldSchema(name='id', dtype=DataType.INT64, descrition='ids', is_primary=True, auto_id=False),
            FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, descrition='embedding vectors', dim=dim),
            FieldSchema(name='category', dtype=DataType.INT64, descrition='category'),
        ]

        schema = CollectionSchema(fields=fields, description='mini imagenet text image search')
        collection = Collection(name=collection_name, schema=schema)

        # 在配置文件中填写索引参数
        index_params = config['milvus']['index_params']

        # 根据字段建立索引
        collection.create_index(field_name="embedding", index_params=index_params)
        return collection

    @staticmethod
    def list_collections():
        utility.list_collections()

    @staticmethod
    def drop_collection(collection_name):
        utility.drop_collection(collection_name)
