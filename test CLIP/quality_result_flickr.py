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
checkpoint_path = "/root/sdusht/data/3_2021-NeurIPS-NCR-main/new_output_model_CLIP/finetune/COCO/0.4/model_best.pth.tar"
checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda(device=device))
model.load_state_dict(checkpoint["model"], strict=False)
model.eval()




