import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoTokenizer

from CUT_turbo import CUT_turbo
from utils.training_utils import UnpairedDataset, build_transform


def parse_args_unpaired_contrastive_inference():
    """
    Parses command-line arguments used for CUT-Turbo inference.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='path to the input image or input image folder')
    parser.add_argument('--pretrained_path', type=str, default=None, help='path to a local model state dict to be used')
    parser.add_argument('--output_dir', type=str, default='output', help='the directory to save the output')
    parser.add_argument('--image_prep', type=str, default='resize_512x512', help='the image preparation method')
    args = parser.parse_args()
    return args


def single_image_inference(args):
    if args.pretrained_path is None:
        raise ValueError("pretrained_path should be provided.")

    # initialize the model
    model = CUT_turbo(args).cuda()
    model.eval()
    model.unet.enable_xformers_memory_efficient_attention()

    T_val = build_transform(args.image_prep)

    input_image = Image.open(args.image_path).convert('RGB')
    with torch.no_grad():
        val_image = T_val(input_image)
        x_t = transforms.ToTensor()(val_image)
        x_t = transforms.Normalize([0.5], [0.5])(x_t).unsqueeze(0).cuda()
        output = model(x_t, caption=args.prompt)

    output_pil = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
    output_pil = output_pil.resize((input_image.width, input_image.height), Image.LANCZOS)

    # save the output image
    bname = os.path.basename(args.image_path)
    os.makedirs(args.output_dir, exist_ok=True)
    output_pil.save(os.path.join(args.output_dir, bname))


if __name__ == '__main__':
    args = parse_args_unpaired_contrastive_inference()
    single_image_inference(args)