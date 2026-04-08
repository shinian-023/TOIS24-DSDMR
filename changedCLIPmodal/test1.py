from transformers import AutoTokenizer, CLIPTextModelWithProjection
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from PIL import Image



image_path = "/root/sdusht/dataset/COCO_images/COCO_val_ims/COCO_val2014_000000000139.jpg"
image = Image.open(image_path).convert("RGB")
text = "A child holding a flowered umbrella and petting a yak ."


model1 = CLIPTextModelWithProjection.from_pretrained("/root/sdusht/huggingface/clip-vit-base-patch32")
tokenizer1 = AutoTokenizer.from_pretrained("/root/sdusht/huggingface/clip-vit-base-patch32")

inputs1 = tokenizer1(text, padding=True, return_tensors="pt")

outputs1 = model1(**inputs1)
text_embeds1 = outputs1.text_embeds




model2 = CLIPVisionModelWithProjection.from_pretrained("/root/sdusht/huggingface/clip-vit-base-patch32")
processor2 = AutoProcessor.from_pretrained("/root/sdusht/huggingface/clip-vit-base-patch32")

inputs2 = processor2(images=image, return_tensors="pt")

outputs2 = model2(**inputs2)
image_embeds2 = outputs2.image_embeds

print()