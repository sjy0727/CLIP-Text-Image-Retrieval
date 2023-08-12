#!/usr/bin/env python
# -*- encoding : utf-8 -*-
"""
@Author :sunjunyi
@Time   :2023/7/29 13:19
"""
import os
import json
import pickle
import yaml

import gradio as gr
import numpy as np
import pandas as pd

from config import config
from model import OnnxModel, HfModel
from redis_handler import RedisHandler
from milvus_handler import MilvusHandler
from metric import compute_mrr
from PIL import Image
from pymilvus import MilvusClient, connections, FieldSchema, CollectionSchema, DataType, Collection, utility

root_dir = './mini-imagenet/'
with open(os.path.join(root_dir, 'classes_name.json'), 'r') as f:
    mini_imagenet_label = json.load(f)

# 文本标签到数字的字典
label2cate = {i[1].replace('_', ' '): i[0] for i in list(mini_imagenet_label.values())}

# 文本标签列表
labels = [i[1].replace('_', ' ') for i in list(mini_imagenet_label.values())]

# 标签描述文件
captions = ['this is a picture of ' + i for i in labels]


# 图像id到图像文件的映射函数
def id2image(img_id):
    root_dir = './mini-imagenet/'
    val_data = pd.read_csv(os.path.join(root_dir, 'new_val.csv'))  # 只对12000张测试集的图像做检索，
    img_path = os.path.join(root_dir, 'images', val_data['filename'][img_id])
    img = Image.open(img_path)
    return img


class QueryService:
    def __init__(self, model_name, use_onnx=config.onnx.use_onnx, use_redis=config.redis.use_redis):
        self.model_name = model_name
        self.use_onnx = use_onnx
        self.use_redis = use_redis

        if self.use_redis:
            self.redis_handler = RedisHandler()

        self.milvus_handler = MilvusHandler()
        self.model = OnnxModel(model_name) if self.use_onnx else HfModel(model_name)

    def __call__(self, query_text, topk, model_name, return_metrics=False):
        # if not self.use_onnx and self.model_name != model_name:
        #     self._reload(model_name)

        if return_metrics:
            recalls, mrrs = self._compute_metrics(query_text)
            return recalls, mrrs
        else:
            ids, distances, categories = self._search_categories(query_text, topk)
            images = list(map(id2image, ids))
            captions = list(map(lambda x: labels[x], categories))
            return list(zip(images, captions))

    # 根据 单个文本向量搜索topk个结果
    def _search_categories(self, query_text, topk):
        if self.use_redis:
            # 如果查询文本是单个字符串
            if type(query_text) == str:
                search_res = self.redis_handler.get(query_text)
            # 如果查询文本是字符串列表
            elif type(query_text) == list:
                search_res = self.redis_handler.redis_client.mget(query_text)
            else:
                return NotImplementedError

            # 如果没在redis中找到结果
            if search_res == None or None in search_res:  # TODO:待优化
                ids, distances, categories = self._embed_and_search(query_text, topk)

                res_pack = list(zip(ids, distances, categories))
                # 如果查询文本是单个字符串则转为列表，方便统一处理
                if type(query_text) == str:
                    query_text = [query_text]
                for text, pack in zip(query_text, res_pack):
                    self.redis_handler.set(text, pickle.dumps(pack))

                if type(query_text) != str:
                    return ids, distances, categories
                else:
                    return ids[0], distances[0], categories[0]
            else:
                if type(query_text) == str:
                    id, distance, category = self.redis_handler.get(query_text)
                    return id, distance, category
                elif type(query_text) == list:
                    deserialize_res = self.redis_handler.mget(query_text)
                    ids, distances, categories = tuple(zip(*deserialize_res))
                    return ids, distances, categories
        # 如果不使用redis
        else:
            ids, distances, categories = self._embed_and_search(query_text, topk)
            if type(query_text) != str:
                return ids, distances, categories
            else:
                return ids[0], distances[0], categories[0]

    def _embed_and_search(self, query_text, topk):
        text_embeds = self.model(text=query_text)
        ids, distances, categories = self.milvus_handler.search(text_embeds, topk)
        return ids, distances, categories

    # 计算相关指标
    def _compute_metrics(self, query_text):
        from sklearn.metrics import precision_score, recall_score
        recalls, mrrs = [], []
        topk_list = [1, 3, 5, 10]
        ids, _, categories = self._search_categories(query_text, max(topk_list))
        for k in topk_list:
            targets = np.array([i for i in range(100)]).repeat(k)
            categories_flat = np.array(categories)[:, :k].flatten()

            recall = recall_score(targets, categories_flat, average='micro')
            mrr = compute_mrr([i for i in range(100)], np.array(categories)[:, :k])

            recalls.append(round(100 * recall, 4))
            mrrs.append(round(100 * mrr, 4))
        return recalls, mrrs


class CalMetrics:
    def __init__(self, query_service):
        self.query_service = query_service

    def __call__(self):
        recalls, mrrs = self.query_service(query_text=labels, topk=10, model_name=self.query_service.model_name,
                                           return_metrics=True)
        return f"""
                |            | **Recall (%)** | **mAP (%)** |
                |:----------:|:--------------:|:-----------:|
                |  **top@1** |{recalls[0]}    |{mrrs[0]}    |
                |  **top@3** |{recalls[1]}    |{mrrs[1]}    |
                |  **top@5** |{recalls[2]}    |{mrrs[2]}    |
                | **top@10** |{recalls[3]}    |{mrrs[3]}    |
                """


def text2image_gr():
    clip = config.gradio.checkpoint_dir
    # blip2 = 'blip2-2.7b'

    title = "<h1 align='center'>多模态大模型图像检索应用</h1>"
    description = '本项目基于mini imagenet数据集微调'

    examples = [
        ["dugong", 10, clip, ],
        ["robin", 10, clip, ],
        ["triceratops", 10, clip, ],
        ["green mamba", 10, clip, ]
    ]

    with gr.Blocks() as demo:
        gr.Markdown(title)
        gr.Markdown(description)
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Column(scale=2):
                    query_text = gr.Textbox(value="house finch", label="请填写搜索文本", elem_id=0, interactive=True)

                # 注意topk < search_param的ef
                topk = gr.components.Slider(minimum=1, maximum=20, step=1, value=10, label="返回图片数",
                                            elem_id=2)

                model_name = gr.components.Radio(label="模型选择", choices=[clip],
                                                 value=clip, elem_id=3)

                btn1 = gr.Button("搜索")

            with gr.Column(scale=100):
                out1 = gr.Gallery(label="检索结果为:", columns=5, height=200)

            with gr.Column(scale=2):
                with gr.Column(scale=6):
                    out2 = gr.Markdown(
                        """
                        |            | **Recall (%)** | **mAP (%)** |
                        |:----------:|:--------------:|:-----------:|
                        |  **top@1** |                |             |
                        |  **top@3** |                |             |
                        |  **top@5** |                |             |
                        | **top@10** |                |             |
                        """
                    )
                btn2 = gr.Button("计算检索100类的平均指标", scale=1)

        inputs = [query_text, topk, model_name]

        gr.Examples(examples, inputs=inputs)

        # TODO: 添加推理时间 查询时间的显示框 datatime库 timeit库
        model_query = QueryService(model_name.value)
        cal_metrics = CalMetrics(model_query)

        btn1.click(fn=model_query, inputs=inputs, outputs=out1)
        btn2.click(fn=cal_metrics, inputs=None, outputs=out2)

    return demo


if __name__ == "__main__":
    with gr.TabbedInterface(
            [text2image_gr()],
            ["文到图搜索"],
    ) as demo:
        demo.launch(
            enable_queue=True,
            server_name='0.0.0.0'
            # share=True
        )
