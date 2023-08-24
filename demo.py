#!/usr/bin/env python
# -*- encoding : utf-8 -*-
"""
@Author :sunjunyi
@Time   :2023/7/29 13:19
"""
import os
import io
import json
import pickle
import base64

import gradio as gr
import numpy as np
import pandas as pd

from config import config
from model import OnnxTextModel, HfTextModel, HfVisionModel
from image_caption_dataset import get_labels_and_cates
from db_handler import MilvusHandler, RedisHandler
from metric import compute_mrr, NDCG, MRR, mAP

from PIL import Image
from pymilvus import MilvusClient, connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# root_dir = './mini-imagenet/'
# with open(os.path.join(root_dir, 'classes_name.json'), 'r') as f:
#     mini_imagenet_label = json.load(f)

# 文本标签到数字的字典
# label2cate = {i[1].replace('_', ' '): i[0] for i in list(mini_imagenet_label.values())}

# 文本标签列表
# labels = [i[1].replace('_', ' ') for i in list(mini_imagenet_label.values())]

# 标签描述文件
# captions = ['this is a picture of ' + i for i in labels]


# 所有图像的标签和类别
labels, cates = get_labels_and_cates(dataset=config.dataset.name, is_train=False)
label2cate = {label: cate for label, cate in zip(labels, cates)}
cate2label = {str(cate): label for label, cate in zip(labels, cates)}
# 外部字典存储图像的id与image base64
extraPairDict = {}


# 图像id到图像文件的映射函数
def id2image(img_id):
    root_dir = './mini-imagenet/'
    val_data = pd.read_csv(os.path.join(root_dir, 'new_val.csv'))  # 只对12000张测试集的图像做检索，
    # 如果img_id不在数据集的标签文件中，则是用户自己上传的
    if img_id >= len(val_data['filename']):
        img_base64 = extraPairDict[str(img_id)]
        img = Image.open(io.BytesIO(base64.b64decode(img_base64))).convert("RGB")
    else:
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
        self.text_model = OnnxTextModel(model_name) if self.use_onnx else HfTextModel(model_name)
        self.vision_model = HfVisionModel(model_name)

    def __call__(self, query_text, topk, model_name):
        ids, distances, categories = self._search_categories(query_text, topk)
        images = list(map(id2image, ids))
        # captions = list(map(lambda x: labels[x], categories))  # 根据类别数字找到对应的描述
        captions = list(map(lambda x: cate2label[str(x)], categories))
        return list(zip(images, captions))

    # 根据 单个文本向量搜索topk个结果
    def _search_categories(self, query_text, topk):
        if self.use_redis:
            # 如果查询文本是单个字符串
            if isinstance(query_text, str):
                search_res = self.redis_handler.get(query_text)
            # 如果查询文本是字符串列表
            elif isinstance(query_text, list):
                search_res = self.redis_handler.redis_client.mget(query_text)
            else:
                return NotImplementedError

            # 如果没在redis中找到结果
            if search_res is None or None in search_res:  # TODO:待优化
                ids, distances, categories = self._embed_and_search(query_text, topk)

                res_pack = list(zip(ids, distances, categories))
                # 如果查询文本是单个字符串则转为列表，方便统一处理
                if isinstance(query_text, str):
                    query_text = [query_text]
                for text, pack in zip(query_text, res_pack):
                    self.redis_handler.set(text, pickle.dumps(pack))

                if not isinstance(query_text, str):
                    return ids, distances, categories
                else:
                    return ids[0], distances[0], categories[0]
            else:
                if isinstance(query_text, str):
                    id, distance, category = self.redis_handler.get(query_text)
                    return id, distance, category
                elif isinstance(query_text, list):
                    deserialize_res = self.redis_handler.mget(query_text)
                    ids, distances, categories = tuple(zip(*deserialize_res))
                    return ids, distances, categories
        # 如果不使用redis
        else:
            ids, distances, categories = self._embed_and_search(query_text, topk)
            if not isinstance(query_text, str):
                return ids, distances, categories
            else:
                return ids[0], distances[0], categories[0]

    def _embed_and_search(self, query_text, topk):
        text_embeds = self.text_model(text=query_text)
        ids, distances, categories = self.milvus_handler.search(text_embeds, topk)
        return ids, distances, categories

    def _PIL2Base64(self, image):
        # 创建一个BytesIO对象，用于临时存储图像数据
        image_data = io.BytesIO()

        # 将图像保存到BytesIO对象中，格式为JPEG
        image.save(image_data, format='JPEG')

        # 将BytesIO对象的内容转换为字节串
        image_data_bytes = image_data.getvalue()

        # 将图像数据编码为Base64字符串
        encoded_image = base64.b64encode(image_data_bytes).decode('utf-8')
        return encoded_image

    def embed_and_insert(self, upload_image, label):
        image_embeds = self.vision_model(images=upload_image)
        text_embeds = self.text_model(text=label)
        ids = [self.milvus_handler.collection.num_entities + 1]
        # 如果label不在字典中
        if label2cate.get(label) is None:
            labels.append(label)
            label2cate[label] = max(cates) + 1
            cates.append(label2cate[label])
            cate2label[str(label2cate[label])] = label
            categories = [label2cate[label]]
            # 插入到id到image base64的字典
            extraPairDict[str(self.milvus_handler.collection.num_entities + 1)] = self._PIL2Base64(upload_image)
        else:
            categories = [label2cate[label]]
        # ids对应的主键序号列表， categories类别对应的数字列表
        insert_datas = [ids, (image_embeds + text_embeds) / 2, categories]  # 插入的数据可以是list[dict] or list[list]
        self.milvus_handler.insert(data=insert_datas)
        return '已上传至图库中'

    # 计算相关指标
    def compute_metrics(self, query_text=labels):
        from sklearn.metrics import recall_score
        recalls = []
        mrrs = []
        ndcgs = []
        maps = []

        topk_list = [1, 3, 5, 10]
        ids, _, categories = self._search_categories(query_text, max(topk_list))

        for k in topk_list:
            # targets = np.array([i for i in range(100)])
            targets = np.array(cates)
            categories_k = np.array(categories)[:, :k]

            targets_repeat = targets.repeat(k)
            categories_flat = categories_k.flatten()

            recall = recall_score(targets_repeat, categories_flat, average='micro')
            # mrr = compute_mrr(targets, categories)
            mrr = MRR(categories_k, targets)
            ndcg = NDCG(categories_k, targets)
            m_ap = mAP(categories_k, targets)

            recalls.append(round(100 * recall, 4))
            # mrrs.append(round(100 * mrr, 4))
            mrrs.append(round(100 * mrr, 4))
            ndcgs.append(round(100 * ndcg, 4))
            maps.append(round(100 * m_ap, 4))
        # return recalls, mrrs, ndcgs, maps
        return f"""
                |            | **Recall (%)** | **MRR (%)** | **NDCG (%)** | **mAP (%)** |
                |:----------:|:--------------:|:-----------:|:------------:|:-----------:|
                |  **top@1** |{recalls[0]}    |{mrrs[0]}    |{ndcgs[0]}    |{maps[0]}    |
                |  **top@3** |{recalls[1]}    |{mrrs[1]}    |{ndcgs[1]}    |{maps[1]}    |
                |  **top@5** |{recalls[2]}    |{mrrs[2]}    |{ndcgs[2]}    |{maps[2]}    |
                | **top@10** |{recalls[3]}    |{mrrs[3]}    |{ndcgs[3]}    |{maps[3]}    |
                """


def text2image_gr(model_query, model_name=config.gradio.checkpoint_dir):
    clip = model_name
    # blip2 = 'blip2-2.7b'

    title = "<h1 align='center'>多模态大模型图像检索应用</h1>"
    description = '本项目基于CLIP与Milvus构建'

    examples = [
        ["dugong", 10, clip],
        ["robin", 10, clip],
        ["triceratops", 10, clip],
        ["green mamba", 10, clip]
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
                        |            | **Recall (%)** | **MRR (%)** | **NDCG (%)** | **mAP (%)** |
                        |:----------:|:--------------:|:-----------:|:------------:|:-----------:|
                        |  **top@1** |                |             |              |             |
                        |  **top@3** |                |             |              |             |
                        |  **top@5** |                |             |              |             |
                        | **top@10** |                |             |              |             |
                        """
                    )
                btn2 = gr.Button("计算检索平均指标", scale=1)

        inputs = [query_text, topk, model_name]

        gr.Examples(examples, inputs=inputs)

        # TODO: 添加推理时间 查询时间的显示框 datatime库 timeit库
        # model_query = QueryService(model_name.value)

        btn1.click(fn=model_query, inputs=inputs, outputs=out1)
        btn2.click(fn=model_query.compute_metrics, inputs=None, outputs=out2)

    return demo


def upload2db_gr(model_query):
    with gr.Blocks() as demo:
        with gr.Row():  # 行
            with gr.Column():  # 列
                img = gr.Image(type='pil')
                # label = gr.Dropdown(labels, label='图像类别')
                label = gr.Textbox(label='图像类别')  # 避免下拉栏太多
                with gr.Row():
                    gr.ClearButton(img)
                    btn = gr.Button("提交")

            with gr.Column():
                md = gr.Markdown()

        btn.click(fn=model_query.embed_and_insert, inputs=[img, label], outputs=md)
    return demo


if __name__ == "__main__":
    model_name = config.gradio.checkpoint_dir
    model_query = QueryService(model_name)

    with gr.TabbedInterface(
            [text2image_gr(model_query, model_name), upload2db_gr(model_query)],
            ['以文搜图', '上传图片'],
    ) as demo:
        demo.launch(
            enable_queue=True,
            server_name='0.0.0.0'
            # share=True
        )
