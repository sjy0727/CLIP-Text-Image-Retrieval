#!/usr/bin/env python
# -*- encoding : utf-8 -*-
"""
@Author :sunjunyi
@Time   :2023/8/6 20:25
"""
import os
import yaml
import torch
import onnxruntime
import numpy as np
from transformers import AutoProcessor, CLIPTextModelWithProjection

# 加载配置文件
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)


class OnnxModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.onnx_path = os.path.join(config['onnx']['save_dir'], model_name.split('/')[1] + '_text_encoder.onnx')
        self.providers = 'CUDAExecutionProvider' if torch.cuda.is_available() else 'CPUExecutionProvider'
        self.session = onnxruntime.InferenceSession(self.onnx_path, providers=[self.providers])
        self.processor = AutoProcessor.from_pretrained(model_name)

    def __call__(self, text):
        text = self.processor(text=text, return_tensors='np', padding=True)
        text_token = dict(text)
        for i in text_token:
            text_token[i] = text_token[i].astype(np.int64)
        text_embeds = self.session.run(None, text_token)[0]
        return text_embeds / np.linalg.norm(text_embeds, axis=1, keepdims=True)


class HfModel:
    def __init__(self, model_name):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.processor = self._load_model_and_processor(model_name)
        self._warmup()

    def _load_model_and_processor(self, model_name):
        model = CLIPTextModelWithProjection.from_pretrained(model_name).to(self.device)
        processor = AutoProcessor.from_pretrained(model_name)
        return model, processor

    def _warmup(self):
        input = self.processor(text='warmup text', return_tensors='pt', padding=True).to(self.device)
        self.model.eval()
        with torch.no_grad():
            self.model(**input)

    @classmethod
    def _empty_cache(self):
        torch.cuda.empty_cache()

    # 重新加载模型和预处理器
    def _reload(self, model_name):
        self._empty_cache()
        self.model_name = model_name
        self.model, self.processor = self._load_model_and_processor(model_name)

    def __call__(self, text):
        text_token = self.processor(text=text, return_tensors='pt', padding=True).to(self.device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(**text_token)
        # 文本embeds向量归一化
        output.text_embeds /= output.text_embeds.norm(dim=-1, keepdim=True)
        text_embeds = output.text_embeds.cpu().numpy()
        return text_embeds
