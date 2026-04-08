import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import requests
from transformers import AutoProcessor, CLIPVisionModel
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextConfig
from transformers import AutoTokenizer, CLIPTextModelWithProjection
from transformers import AutoProcessor, CLIPVisionModelWithProjection


class multimodal_encoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, d_ff, dropout):
        super(multimodal_encoder, self).__init__()

        self.n_layers = n_layers

        self.layers = nn.ModuleList([
            multimodal_encoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt

        for i in range(self.n_layers):
            output = self.layers[i](
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )

        return output



class multimodal_encoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(multimodal_encoderLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Self-attention layer
        tgt2 = self.norm1(tgt)
        tgt2, _ = self.self_attn(tgt2, tgt2, tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout(tgt2)

        # Multi-head attention layer
        tgt2 = self.norm2(tgt)
        tgt2, _ = self.multihead_attn(tgt2, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout(tgt2)

        # Feedforward layer
        tgt2 = self.norm3(tgt)
        tgt2 = F.relu(self.linear1(tgt2))
        tgt2 = self.dropout(tgt2)
        tgt2 = self.linear2(tgt2)
        tgt = tgt + self.dropout(tgt2)

        return tgt

# # CLIPVisionModel
# visual_encoder = CLIPVisionModel.from_pretrained("/root/sdusht/huggingface/clip-vit-base-patch32")
# processor = AutoProcessor.from_pretrained("/root/sdusht/huggingface/clip-vit-base-patch32")
# # inputs1 = processor1(images=image, return_tensors="pt")
# # outputs1 = model1(**inputs1)
# # last_hidden_state1 = outputs1.last_hidden_state
# # pooled_output1 = outputs1.pooler_output  # pooled CLS states
#
#
# # CLIPTextModel
# text_encoder = CLIPTextModel.from_pretrained("/root/sdusht/huggingface/clip-vit-base-patch32")
# tokenizer = AutoTokenizer.from_pretrained("/root/sdusht/huggingface/clip-vit-base-patch32")
# # inputs2 = tokenizer2(["A child holding a flowered umbrella and petting a yak ."], padding=True, return_tensors="pt")
# # outputs2= model2(**inputs2)
# # last_hidden_state2 = outputs2.last_hidden_state
# pooled_output2 = outputs2.pooler_output  # pooled (EOS token) states







