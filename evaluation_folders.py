import os
import argparse
import numpy as np
import torch
from PIL import Image
from cleanfid.fid import build_feature_extractor, compare_folders
from tqdm import tqdm

from utils.dino_struct import DinoStructureLoss
from utils.training_utils import build_transform

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_path', type=str, required=True, help='path to the generated images')
    parser.add_argument('--input_path', type=str, required=True, help='path to the input images')
    parser.add_argument('--ref_path', type=str, required=True, help='path to the reference images')
    parser.add_argument("--val_img_prep", default="resized_crop_512", type=str)
    args = parser.parse_args()
    T_val = build_transform(args.val_img_prep)
    # compute FID and DINO-Struct scores
    l_dino_scores = []
    net_dino = DinoStructureLoss()
    # cal dino score
    for filename in tqdm(os.listdir(args.input_path)):
        if filename.endswith(('jpg','png','jpeg','bmp')):
            input_file_path = os.path.join(args.input_path, filename)
            input_img = T_val(Image.open(input_file_path).convert('RGB'))
            gen_file_path = os.path.join(args.gen_path, filename)
            assert os.path.exists(gen_file_path), "The generated image corresponding to the input image doesn't exist."
            gen_img = T_val(Image.open(gen_file_path).convert('RGB'))
            dino_ssim = net_dino.calculate_global_ssim_loss(
                net_dino.preprocess(input_img).unsqueeze(0).cuda(),
                net_dino.preprocess(gen_img).unsqueeze(0).cuda()
            ).item()
            l_dino_scores.append(dino_ssim)
    dino_score = np.mean(l_dino_scores)
    del net_dino
    # cal fid
    feat_model = build_feature_extractor("clean", "cuda", use_dataparallel=False)
    fid_score = compare_folders(args.ref_path, args.gen_path, feat_model=feat_model, mode="clean", num_workers=0,
                    batch_size=8, device=torch.device("cuda"), verbose=True,
                    custom_image_tranform=None, custom_fn_resize=None)
    print(f"fid={fid_score:.3f}, dino={dino_score:.3f}")




