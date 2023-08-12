#!/usr/bin/env python
# -*- encoding : utf-8 -*-
"""
@Author :sunjunyi
@Time   :2023/8/5 19:16
"""

import pickle
import redis
from config import config


# 加载配置文件
# with open('config.yaml', 'r', encoding='utf-8') as f:
#     config = yaml.safe_load(f)


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
