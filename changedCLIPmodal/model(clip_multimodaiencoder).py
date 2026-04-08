import torch
from torch import nn, optim
from transformers import CLIPVisionModel, CLIPTextModel, CLIPVisionConfig,  CLIPTextConfig
from changedCLIPmodal.multimodal_encoder import multimodal_encoder


class mm_model(nn.Module):
    def __init__(self):
        super(mm_model, self).__init__()
        self.visual_encoder = CLIPVisionModel(CLIPVisionConfig()).from_pretrained("/data/shihaitao/huggingface/clip-vit-base-patch32")
        # self.processor = AutoProcessor.from_pretrained(CLIP_model)

        self.text_encoder = CLIPTextModel(CLIPTextConfig()).from_pretrained("/data/shihaitao/huggingface/clip-vit-base-patch32")
        # self.tokenizer = AutoTokenizer.from_pretrained(CLIP_model)

        self.multimodal_image_encoder = multimodal_encoder(512, 8, 6, 2048, 0.1)
        self.multimodal_text_encoder = multimodal_encoder(512, 8, 6, 2048, 0.1)

        self.image_projection = nn.Linear(768, 512)
        self.text_projection = nn.Linear(512, 512)

    def forward(self, images, texts):
        # input_image = self.processor(images=image, return_tensors="pt")
        output_image = self.visual_encoder(images)
        output_image_embedding = output_image.last_hidden_state
        output_image_embedding = self.image_projection(output_image_embedding)
        image_query_embedding = output_image_embedding[:,0,:].unsqueeze(1)
        # image_query_embedding_all = torch.zeros(size=(200,50,512))
        # for i in range(200):
        #     x = image_query_embedding[i].tolist()
        #     x = x*50
        #     image_query_embedding_all[i] = torch.tensor(x)

        # input_text = self.tokenizer(text, padding=True, return_tensors="pt")
        output_text = self.text_encoder(texts)
        output_text_embedding = output_text.last_hidden_state
        output_text_embedding = self.text_projection(output_text_embedding)
        text_query_embedding = output_text_embedding[:,-1,:].unsqueeze(1)
        # text_query_embedding_all = torch.zeros(size=(200,77,512))
        #
        # for i in range(200):
        #     x = text_query_embedding[i].tolist()
        #     x = x*77
        #     text_query_embedding_all[i] = torch.tensor(x)
        image_query_embedding, text_kv_embedding = [x.transpose(1, 0) for x in (image_query_embedding, output_text_embedding)]
        text_query_embedding, image_kv_embedding = [x.transpose(1, 0) for x in (text_query_embedding, output_image_embedding)]
        # image_query_embedding_all, output_text_embedding = [x.transpose(1, 0) for x in (image_query_embedding_all, output_text_embedding)]
        # text_query_embedding_all, output_image_embedding = [x.transpose(1, 0) for x in (text_query_embedding_all, output_image_embedding)]

        multimodal_image_output = self.multimodal_image_encoder(image_query_embedding, text_kv_embedding)
        multimodal_text_output = self.multimodal_text_encoder(text_query_embedding, image_kv_embedding)

        multimodal_image_feature = multimodal_image_output.squeeze(0)
        multimodal_text_feature = multimodal_text_output.squeeze(0)

        logits_per_image = multimodal_image_feature @ multimodal_text_feature.T
        logits_per_text = multimodal_text_feature @ multimodal_image_feature.T
        return logits_per_image, logits_per_text

    def encode_image(self, images):
        output_image = self.visual_encoder(images)
        output_image_embedding = output_image.last_hidden_state
        output_image_embedding = self.image_projection(output_image_embedding)
        unimodal_image_feature = output_image_embedding[:, 0, :]
        # image_query_embedding = output_image_embedding[:, 0, :].unsqueeze(1)
        #
        # output_text = self.text_encoder(texts)
        # output_text_embedding = output_text.last_hidden_state
        # output_text_embedding = self.text_projection(output_text_embedding)
        #
        # image_query_embedding, text_kv_embedding = [x.transpose(1, 0) for x in
        #                                             (image_query_embedding, output_text_embedding)]
        # multimodal_image_output = self.multimodal_image_encoder(image_query_embedding, text_kv_embedding)
        # # multimodal_text_output = self.multimodal_text_encoder(text_query_embedding, image_kv_embedding)
        #
        # multimodal_image_feature = multimodal_image_output.squeeze(0)
        #
        return unimodal_image_feature

    def encode_text(self, texts):
        output_text = self.text_encoder(texts)
        output_text_embedding = output_text.last_hidden_state
        output_text_embedding = self.text_projection(output_text_embedding)
        unimodal_text_feature = output_text_embedding[:, -1, :]

        # output_text = self.text_encoder(texts)
        # output_text_embedding = output_text.last_hidden_state
        # output_text_embedding = self.text_projection(output_text_embedding)
        # text_query_embedding = output_text_embedding[:, -1, :].unsqueeze(1)
        #
        # text_query_embedding, image_kv_embedding = [x.transpose(1, 0) for x in
        #                                             (text_query_embedding, output_image_embedding)]
        # multimodal_text_output = self.multimodal_text_encoder(text_query_embedding, image_kv_embedding)
        # multimodal_text_feature = multimodal_text_output.squeeze(0)

        # return unimodal_text_feature, multimodal_text_feature
        return unimodal_text_feature
