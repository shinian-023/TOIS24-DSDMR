import os
import sys
import numpy as np
import random
import torch
import argparse
import time
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# sys.path.append(os.path.join(os.path.dirname(__file__), 'RetinaFace'))
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
#sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from CLIPFinetune.train_model_addGMM_tripletloss_dynamic_margin import main

# current_time
current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
# Hyper Parameters
parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
parser.add_argument(
        "--lr_update",
        default=10,
        type=int,
        help="Number of epochs to update the learning rate.",
)
parser.add_argument(
    "--learning_rate",
    default= 5e-7,
    help="Learning_rate of the model."
)
parser.add_argument(
    "--data_name",
    default="COCO",
    help="Name of the daataset{flickr, COCO}.",
)
parser.add_argument(
    "--noisy_file",
    default="/root/sdusht/dataset/NCR-data/noise_index_DECL/noise_file/coco/noise_inx_0.4.npy",
    help="path to noisy file."
)

# ----------------------- training setting ----------------------#
parser.add_argument(
    "--batch_size",
    default=64,
    type=int,
    help="Size of a training mini-batch."
)
parser.add_argument(
    "--num_epochs",
    default=15,
    type=int,
    help="Number of training epochs."
)
parser.add_argument(
    "--output_dir",
    default="/root/sdusht/data/3_2021-NeurIPS-NCR-main/new_output_model_CLIP/fineturn_0P5_sepration/COCO/0.4/tripletloss_dynamic_margin_batch64",
    help="Output dir."
)
parser.add_argument(
    "--margin_image",
    default=0.3,
    type=float,
    help="Rank loss margin of image."
)
parser.add_argument(
    "--margin_text",
    default=0.2,
    type=float,
    help="Rank loss margin of text."
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run():
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    Resume = False
    if Resume:
        checkpoint_path = ''
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda(device=device))
        opt = checkpoint["opt"]
        print("\n*-------- Experiment Config --------*")
        print(opt)
        main(device, opt, checkpoint=checkpoint)
    else:
        # load arguments
        opt = parser.parse_args()
        print("\n*-------- Experiment Config --------*")
        print(opt)

        main(device, opt)


if __name__ == '__main__':
    run()