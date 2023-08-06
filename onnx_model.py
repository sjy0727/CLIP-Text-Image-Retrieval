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
from transformers import AutoProcessor

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
