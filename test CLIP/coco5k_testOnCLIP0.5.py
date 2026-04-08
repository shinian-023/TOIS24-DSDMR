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
torch.cuda.set_device(2)
model, preprocess = clip.load("ViT-B/32", device=2)
checkpoint_path = "/root/sdusht/data/3_2021-NeurIPS-NCR-main/output_finetune_CLIP_0.2/COCO_model_para7_8.pkl"
checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda(2))
model.load_state_dict(checkpoint)
model.cuda().eval()


# 读数据的路径
data_path = "/root/sdusht/dataset/NCR-data/data/coco_precomp"
vocab_path = "/root/sdusht/dataset/NCR-data/vocab"
data_split = "testall"

picturePath = "/root/sdusht/dataset/COCO_images/COCO_test_ims"
sortedImageName = "/root/sdusht/dataset/COCO_test_names.txt"


# 获取 coco5k_tesk 的所有数据
texts = []
images = []

 # 获得coco5k_test captions
with open(os.path.join(data_path, "%s_caps.txt" % data_split), "r", encoding="utf-8") as f:
    for line in f:
        texts.append(line.strip())

 # 获得coco5k_test images(5000)
f = open(sortedImageName, "r")
sortedImageList = f.readlines()
for i in range(len(sortedImageList)):
    sortedImageList[i] = sortedImageList[i].strip('\n')

for filename in sortedImageList:
    image = Image.open(os.path.join(picturePath, filename)).convert("RGB")
    images.append(preprocess(image))
print("数据读取完毕,images:{},texts:{}".format(len(images), len(texts)))

# 构建 图像和文本的特征
# 由于 model.encode_image() 和 model.encode_text() 比较占内存，
# 因此，数据处理成特征的过程需要循环处理，每次处理 1000 个数据，否则会造成 out of memory
zero_image = np.zeros((1000, 512))
zero_text = np.zeros((5000, 512))
image_features_all = torch.tensor(zero_image, dtype=torch.float32, device=2)
text_features_all = torch.tensor(zero_text, dtype=torch.float32, device=2)
for i in range(5):
    image_input = torch.tensor(np.stack(images[1000*i:1000*(i+1)])).cuda()
    text_tokens = clip.tokenize([text for text in texts[5000*i:5000*(i+1)]]).cuda()
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(text_tokens).float()
    image_features_all = torch.cat((image_features_all, image_features), dim=0)
    text_features_all = torch.cat((text_features_all, text_features), dim=0)
    print("{} *1000 data features over".format((i + 1)))
image_features_all = image_features_all[1000:]
text_features_all = text_features_all[5000:]
print("数据特征提取完毕,image_features:{}, text_features:{}".format(len(image_features_all), len(text_features_all)))

# 计算余弦相似度
image_features_all /= image_features_all.norm(dim=-1, keepdim=True)
text_features_all /= text_features_all.norm(dim=-1, keepdim=True)
similarity = image_features_all.cpu().numpy()@text_features_all.cpu().numpy().T

# bi-directional retrieval
r, rt = i2t(5000, similarity, 5, return_ranks=True)
ri, rti = t2i(5000, similarity, 5, return_ranks=True)

ar = (r[0] + r[1] + r[2]) / 3
ari = (ri[0] + ri[1] + ri[2]) / 3
rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
print("rsum: %.1f" % rsum)
print("Average i2t Recall: %.1f" % ar)
print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
print("Average t2i Recall: %.1f" % ari)
print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)