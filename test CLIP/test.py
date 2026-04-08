# import torch
# from PIL import Image
#
# from lavis.models import load_model_and_preprocess
# from lavis.processors import load_processor
import numpy as np

x = np.array([1,2,3,1,5,6,7,8,9,1])

a = np.where(x < 2)[0]
print()