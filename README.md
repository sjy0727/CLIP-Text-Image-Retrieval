# 基于CLIP的以文搜图

## 项目概述
CLIP 图文检索项目是一个基于 OpenAI 的 CLIP 模型的应用。该项目旨在通过输入文本描述来检索与之相匹配的图片。它结合了 CLIP 模型的强大图像和文本理解能力，使得用户可以通过简单的文本查询来发现感兴趣的图片。为了提高检索效率，项目采用了 `Milvus` 和 `Redis` 数据库，并利用 `ONNX` 进行推理加速。

## 主要功能

1. 图文检索：用户可以输入文本描述，系统将利用 CLIP 模型从数据库中检索出与之最匹配的图片，并返回结果。

2. 数据库存储：项目使用 Milvus 数据库来存储图像特征向量，利用其高效的相似度搜索功能进行图像检索。

3. 缓存优化：通过 Redis 数据库，将一部分检索结果缓存在内存中，以加速后续的相似度搜索。

4. ONNX 推理加速：通过将 CLIP 模型转换为 ONNX 格式，利用 ONNX Runtime 进行推理加速，提高检索的实时性。
## 安装指南

在这里提供项目的安装指南：

1. 下载安装单机版`milvus`向量数据库 https://milvus.io/docs/install_standalone-docker.md
2. 安装所需依赖项：

```bash
$ git clone xxxx
$ cd xxxx
$ pip install -r requirements.txt
```

## 数据集准备

- 下载`mini-imagenet`数据集
- 链接 https://pan.baidu.com/s/1Uro6RuEbRGGCQ8iXvF2SAQ password: hl31

### 目录结构

把*.ipynb文件与mini-imagenet文件夹放在同级目录下

```
mini-imagenet
├── classes_name.json
├── imagenet_class_index.json
├── images
│   ├── xxx.jpg
│   ├── xxx.jpg
│   └── ...
├── new_train.csv
├── new_val.csv
├── train.json
└── val.json
```

2016年google
DeepMind团队从Imagnet数据集中抽取的一小部分（大小约3GB）制作了Mini-Imagenet数据集，共有100个类别，每个类别都有600张图片，共60000张（都是.jpg结尾的文件），而且图像的大小并不是固定的

## 微调

- 通过`Huggingface`的`accelerate`实现单机多卡训练

```bash
$ accelerate launch --multi_gpu \
                    --mixed_precision=fp16 \
                    --num_processes=2 \
                    clip_finetuning.py \
                    --ep=5 \
                    --lr=5e-5 \
                    --bs=64
```

## 部署

- 启动`Grdio`界面

```bash
$ python demo.py
```

## 用例示例





