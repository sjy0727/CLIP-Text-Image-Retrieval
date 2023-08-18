#!/usr/bin/env python
# -*- encoding : utf-8 -*-
"""
@Author :sunjunyi
@Time   :2023/8/11 23:04
"""


class MilvusConfig:
    def __init__(self):
        self.host = '127.0.0.1'
        self.port = '19530'
        self.collection_name = 'base_patch32_add'
        self.vector_dim = 512
        self.topk = 10

        self.index_params = {
            'metric_type': 'IP',  # 内积距离
            'index_type': 'HNSW',  # 算法类型
            'params': {
                'M': 8,
                'efConstruction': 128
            }
        }

        self.search_params = {
            "metric_type": 'IP',
            "params": {
                "ef": 20
            }  # topk < search_param的ef
        }


class RedisConfig:
    def __init__(self):
        self.use_redis = False
        self.host = '127.0.0.1'
        self.port = 6379
        self.db = 0
        self.expire_time = 3600  # 过期时间


class GradioConfig:
    def __init__(self):
        self.checkpoint_dir = 'openai/clip-vit-base-patch32'


class FinetuneConfig:
    def __init__(self):
        self.checkpoint_dir = 'openai/clip-vit-base-patch32'  # 可以是huggingface上的模型名称或本地文件夹
        self.save_dir = './checkpoint'


class OnnxConfig:
    def __init__(self):
        self.use_onnx = True
        self.checkpoint_dir = 'openai/clip-vit-base-patch32'  # 可以是huggingface上的模型名称或本地文件夹
        self.save_dir = './onnx'


class Config:
    def __init__(self):
        self.milvus = MilvusConfig()
        self.redis = RedisConfig()
        self.gradio = GradioConfig()
        self.finetune = FinetuneConfig()
        self.onnx = OnnxConfig()


config = Config()
