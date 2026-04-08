import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPTextModel, CLIPVisionConfig,  CLIPTextConfig
from changedCLIPmodal.multimodal_encoder import multimodal_encoder
from transformers import CLIPModel, CLIPConfig

class mm_model(nn.Module):
    def __init__(self,
                 pretrained_CLIP_model,
                 ):
        super(mm_model, self).__init__()
        self.CLIP_model = CLIPModel.from_pretrained(pretrained_CLIP_model)

    def forward(self, mode, device, **input):
        if mode =="train":
            device = device
            CLIP_model_output = self.CLIP_model(**input)
            logits_per_image_CLIP = CLIP_model_output.logits_per_image
            logits_per_text_CLIP = CLIP_model_output.logits_per_text


            return logits_per_image_CLIP, logits_per_text_CLIP


        elif mode =="val" or mode =="test" or mode =="filter":
            CLIP_model_output = self.CLIP_model(**input)
            logits_per_image_CLIP = CLIP_model_output.logits_per_image
            logits_per_text_CLIP = CLIP_model_output.logits_per_text
            return logits_per_image_CLIP, logits_per_text_CLIP

