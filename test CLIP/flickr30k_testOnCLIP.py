import numpy as np
import torch
import clip
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from collections import OrderedDict
import torch
from NCR.evaluation import i2t, t2i
from NCR.vocab import Vocabulary, deserialize_vocab
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"

# 创建模型

torch.cuda.set_device(0)
model, preprocess = clip.load("ViT-B/32")
pkl_path = '/root/sdusht/data/6_train-CLIP-FT-main/lightning_logs/version_0/checkpoints/epoch=31-step=1791.ckpt'
ckpt_dict = torch.load(pkl_path)
model.load_state_dict(ckpt_dict['state_dict'])
model.cuda().eval()

# 读数据的路径
data_path = "/root/sdusht/dataset/NCR-data/data/f30k_precomp"
vocab_path = "/root/sdusht/dataset/NCR-data/vocab"
data_split = "test"

picturePath = "/root/sdusht/dataset/flickr30k_images/flickr30k_test_ims"
sortedImage_idsPath = "/root/sdusht/dataset/sorted_flickr30K_test_ids.txt"


# 获取 flickr30k_test_ims 的所有数据
texts = []
images = []

 # 获得flickr30k_test captions
with open(os.path.join(data_path, "%s_caps.txt" % data_split), "r", encoding="utf-8") as f:
    for line in f:
        texts.append(line.strip())

 # 获得flickr30k_test images(5000)
f = open(sortedImage_idsPath, "r")
sortedImageFileName = f.readlines()
for i in range(len(sortedImageFileName)):
    sortedImageFileName[i] = sortedImageFileName[i].strip('\n')

for filename in sortedImageFileName:
    image = Image.open(os.path.join(picturePath, filename)).convert("RGB")
    images.append(preprocess(image))
print("数据读取完毕,images:{},texts:{}读取完毕！ ".format(len(images), len(texts)))

# 构建 图像和文本的特征
image_input = torch.tensor(np.stack(images)).cuda()
text_tokens = clip.tokenize([ text for text in texts]).cuda()

with torch.no_grad():
    image_features = model.encode_image(image_input).float()
    text_features = model.encode_text(text_tokens).float()
print("数据特征提取完毕,image_features:{}, text_features:{}".format(len(image_features), len(text_features)))

# 计算余弦相似度
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = image_features.cpu().numpy()@text_features.cpu().numpy().T
print(similarity)

# bi-directional retrieval
r, rt = i2t(1000, similarity, 5, return_ranks=True)
ri, rti = t2i(1000, similarity, 5, return_ranks=True)

ar = (r[0] + r[1] + r[2]) / 3
ari = (ri[0] + ri[1] + ri[2]) / 3
rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
print("rsum: %.1f" % rsum)
print("Average i2t Recall: %.1f" % ar)
print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
print("Average t2i Recall: %.1f" % ari)
print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)

