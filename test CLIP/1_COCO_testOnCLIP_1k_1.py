#-*-coding:utf8-*
from PIL import Image
import torch
import os
import sys
import json
import time
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 创建模型
pretrained_model_name = "/root/sdusht/huggingface/clip-vit-base-patch32"
model = mm_model(pretrained_model_name)
model.to(device)
processor = AutoProcessor.from_pretrained(pretrained_model_name)
checkpoint_path = "/root/sdusht/data/3_2021-NeurIPS-NCR-main/new_output_model_CLIP/fineturn_0P5_sepration/COCO/0.4/tripletloss_dynamic_margin_batch64/model_best.pth.tar"
checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda(device=device))
model.load_state_dict(checkpoint["model"], strict=False)
model.eval()


def load_data(data_split, data_name):
    if data_split == "test":
        if data_name =="COCO":
            with open('/root/sdusht/data/3_2021-NeurIPS-NCR-main/CLIPFinetune/data/COCO_testall.json', 'r', encoding='utf8') as fp:
                df = json.load(fp)

    return df

test_data = load_data("test", "COCO")
text_inputs = test_data["caption"]
image_inputs = []
for img in range(len(test_data["image"])):
    image_inputs.append(Image.open(test_data["image"][img]).convert("RGB"))

print("begining test!")
result = []  # 5fold cross-validation, only for MSCOCO
j_sim = np.zeros((200, 1000))
i_sim = np.zeros((200, 5000))
for idx in range(5):
    text_input = text_inputs[idx * 5000: (idx + 1) * 5000]
    image_input = image_inputs[idx * 1000: (idx + 1) * 1000]
    for i in range(5):
        image_start = 200 * i
        image_end = 200 * (i + 1)

        for j in range(5):
            text_start = 1000 * j
            text_end = 1000 * (j + 1)
            model_input = processor(text=text_input[text_start: text_end], images=image_input[image_start: image_end],
                                    return_tensors="pt", padding=True)
            model_input.to(device)
            with torch.no_grad():
                logits_per_image, logits_per_text = model(**model_input)
                sim = logits_per_image.cpu().numpy().tolist()
                j_sim = np.concatenate((j_sim.tolist(), sim), axis=1)
                # torch.cuda.empty_cache()

        j_sim = j_sim[: , 1000:]
        i_sim = np.concatenate((i_sim.tolist(), j_sim), axis=0)
        j_sim = np.zeros((200, 1000))

    similarity = i_sim[200: , :]
    i_sim = np.zeros((200, 5000))
    r, rt0 = i2t(
        1000, similarity, per_captions=5, return_ranks=True)
    print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
    ri, rti0 = t2i(
        1000, similarity, per_captions=5, return_ranks=True)
    print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)

    if idx == 0:
        rt, rti = rt0, rti0
    ar = (r[0] + r[1] + r[2]) / 3
    ari = (ri[0] + ri[1] + ri[2]) / 3
    rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
    print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
    result += [list(r) + list(ri) + [ar, ari, rsum]]

print("-----------------------------------")
print("Mean metrics: ")
mean_metrics = tuple(np.array(result).mean(axis=0).flatten())
print("rsum: %.1f" % (mean_metrics[10] * 6))
mean_i2t = (mean_metrics[0] + mean_metrics[1] + mean_metrics[2]) / 3
print("Average i2t Recall: %.1f" % mean_i2t)
print("Image to text: %.1f %.1f %.1f %.1f %.1f" % mean_metrics[:5])
mean_t2i = (mean_metrics[5] + mean_metrics[6] + mean_metrics[7]) / 3
print("Average t2i Recall: %.1f" % mean_t2i)
print("Text to image: %.1f %.1f %.1f %.1f %.1f" % mean_metrics[5:10])

