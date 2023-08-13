#!/usr/bin/env python
# -*- encoding : utf-8 -*-
"""
@Author :sunjunyi
@Time   :2023/8/11 21:06
"""
import pickle
import redis
from config import config
from pymilvus import MilvusClient, connections, FieldSchema, CollectionSchema, DataType, Collection, utility


class MilvusHandler:
    def __init__(self,
                 host=config.milvus.host,
                 port=config.milvus.port,
                 collection_name=config.milvus.collection_name
                 ):
        # 连接向量数据库并加载到内存中
        self.collection_name = collection_name
        connections.connect(host=host, port=port)
        # 连接对应的collection
        self._connect_collection(self.collection_name)

    def _connect_collection(self, collection_name):
        if utility.has_collection(collection_name):  # 如果数据库存在则删除
            self.collection = Collection(collection_name)
            self.collection.load()
        else:
            print('milvus中没有' + collection_name + '对应的collection')

    def search(self, embeds, topk):
        res = self.collection.search(  # TODO: 把milvus部分代码抽离成单个类
            data=embeds,
            anns_field='embedding',
            param=config.milvus.search_params,
            limit=topk,
            output_fields=['category']
        )

        ids = [list(hits.ids) for hits in res]
        distances = [list(hits.distances) for hits in res]
        categories = [[hit.entity.get('category') for hit in hits] for hits in res]
        return ids, distances, categories

    def insert(self, data):  # 可以传list(dict) or list(list)
        return self.collection.insert(data=data)

    def load_and_flush(self):
        self.collection.load()
        self.collection.flush()

    def get_num_entities(self):
        print('已插入' + self.collection.num_entities + '条数据')

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
        print(utility.list_collections())

    @staticmethod
    def drop_collection(collection_name):
        utility.drop_collection(collection_name)
        print('已删除' + collection_name)


class RedisHandler:
    def __init__(self,
                 host=config.redis.host,
                 port=config.redis.port,
                 db=config.redis.db
                 ):

        self.redis_client = redis.StrictRedis(host=host, port=port, db=db)

    def set(self, key, value, ex=config.redis.expire_time):
        self.redis_client.set(key, value, ex)

    def get(self, key):
        result = self.redis_client.get(key)
        if result:
            return pickle.loads(result)
        else:
            return None

    def mget(self, keys):
        deserialize_res = []
        results = self.redis_client.mget(keys)
        for result in results:
            if result:
                deserialize_res.append(pickle.loads(result))
            else:
                deserialize_res.append(None)
        return deserialize_res

    def update_data(self, key, new_value):
        self.redis_client.set(key, new_value)

    def delete_data(self, key):
        self.redis_client.delete(key)


if __name__ == "__main__":
    redis_handler = RedisHandler()

    key = "example_key"
    value = "example_value"

    # 添加数据
    redis_handler.set(key, value)
    print(f"添加数据：{key} -> {value}")

    # 获取数据
    result = redis_handler.get_data(key)
    print(f"获取数据：{key} -> {result}")

    # 更新数据
    new_value = "updated_value"
    redis_handler.update_data(key, new_value)
    print(f"更新数据：{key} -> {new_value}")

    # 获取更新后的数据
    result = redis_handler.get_data(key)
    print(f"获取数据：{key} -> {result}")

    # 删除数据
    redis_handler.delete_data(key)
    print(f"删除数据：{key}")

    # 再次获取数据（已删除）
    result = redis_handler.get_data(key)
    print(f"获取数据：{key} -> {result}")
