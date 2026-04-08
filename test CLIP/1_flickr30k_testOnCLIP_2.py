#-*-coding:utf8-*
from PIL import Image
import torch
import os
import sys
import json
from torch import nn
import numpy as np
from transformers import AutoProcessor

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from NCR.evaluation import i2t, t2i
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from changedCLIPmodal.model1 import mm_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CLIPFinetune.train_model import load_data

from transformers import logging
logging.set_verbosity_error()

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 创建模型
pretrained_model_name = "/root/sdusht/huggingface/clip-vit-base-patch32"
model = mm_model(pretrained_model_name)
model.to(device)
processor = AutoProcessor.from_pretrained(pretrained_model_name)
checkpoint_path = "/root/sdusht/data/3_2021-NeurIPS-NCR-main/new_output_model_CLIP/finetune/flickr/0.2/dataloader_shuffle/model_best.pth.tar"
checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda(device=device))
model.load_state_dict(checkpoint["model"], strict=False)
model.eval()


def load_data(data_split, data_name):
    if data_split == "test":
        if data_name =="flickr":
            with open('/root/sdusht/code/3_2021-NeurIPS-NCR-main/CLIPFinetune/data/f30k_test.json', 'r', encoding='utf8') as fp:
                df = json.load(fp)

    return df

test_data = load_data("test", "flickr")
text_input = test_data["caption"]
image_input = []
for idx in range(len(test_data["image"])):
    image_input.append(Image.open(test_data["image"][idx]).convert("RGB"))

print("begining test!")
j_sim = np.zeros((100, 500))
i_sim = np.zeros((100, 5000))

for i in range(10):
    image_start = 100 * i
    image_end = 100 * (i + 1)

    for j in range(10):
        text_start = 500 * j
        text_end = 500 * (j + 1)
        model_input = processor(text=text_input[text_start: text_end], images=image_input[image_start: image_end],
                                return_tensors="pt", padding=True)
        model_input.to(device)
        with torch.no_grad():
            logits_per_image, logits_per_text = model(**model_input)
            sim = logits_per_image.cpu().numpy().tolist()
            j_sim = np.concatenate((j_sim.tolist(), sim), axis=1)
            # torch.cuda.empty_cache()

    j_sim = j_sim[: , 500:]
    i_sim = np.concatenate((i_sim.tolist(), j_sim), axis=0)
    j_sim = np.zeros((100, 500))

similarity = i_sim[100: , :]

# bi-directional retrieval
r, rt = i2t(1000, similarity, 5, return_ranks=True)
ri, rti = t2i(1000, similarity, 5, return_ranks=True)

ar = (r[0] + r[1] + r[2]) / 3
ari = (ri[0] + ri[1] + ri[2]) / 3
rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
print("test result: ")
print("rsum: %.1f" % rsum)
print("Average i2t Recall: %.1f" % ar)
print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
print("Average t2i Recall: %.1f" % ari)
print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)

