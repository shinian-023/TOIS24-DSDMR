import numpy as np
import torch
import clip
import os
import skimage
import IPython.display
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from collections import OrderedDict
import torch


# 创建模型
model, preprocess = clip.load("ViT-B/32")
model.cuda(3).eval()


# 读数据的路径
data_path = "/root/sdusht/dataset/NCR-data/data/coco_precomp"
vocab_path = "/root/sdusht/dataset/NCR-data/vocab"
data_split = "testall"

picturePath = "/root/sdusht/dataset/coco5k_test_ims"
sortedImage_idsPath = "/root/sdusht/dataset/sorted_coco5K_ids.txt"


# 获取 coco5k_tesk 的所有数据
texts = []
images = []

 # 获得coco5k_test captions
with open(os.path.join(data_path, "%s_caps.txt" % data_split), "r", encoding="utf-8") as f:
    for line in f:
        texts.append(line.strip())

 # 获得coco5k_test images(5000 ——> 25000)
f = open(sortedImage_idsPath, "r")
sortedIdsList = f.readlines()
for i in range(len(sortedIdsList)):
    sortedIdsList[i] = sortedIdsList[i].strip('\n')

sortedImageFileName = []
for i in range(len(sortedIdsList)):
    filename = "COCO_val2014_000000" + sortedIdsList[i] + ".jpg"
    sortedImageFileName.append(filename)
    i = i+1
for filename in sortedImageFileName:
    image = open(os.path.join(picturePath, filename)).convert("RGB")
    images.append(preprocess(image))


# 构建 图像和文本的特征
image_input = torch.tensor(np.stack(images)).cuda(3)
text_tokens = clip.tokenize([ text for text in texts]).cuda(3)

with torch.no_grad():
    image_features = model.encode_image(image_input).float()
    text_features = model.encode_text(text_tokens).float()


# 计算余弦相似度
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T