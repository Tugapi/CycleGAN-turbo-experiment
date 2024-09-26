import numpy as np
from PIL import Image
import cv2
import torch


def canny_from_pil(image, low_threshold=100, high_threshold=200):
    image = np.array(image)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    control_image = Image.fromarray(image)
    return control_image


def canny_from_tensor(images, low_threshold=100, high_threshold=200):
    edges_list = []
    for image in images:
        image_numpy = image.permute(1, 2, 0).numpy()
        edge = cv2.Canny(image_numpy.astype(np.uint8), low_threshold, high_threshold)
        edge = edge[None, :, :]
        edge = np.concatenate([edge, edge, edge], axis=0)
        edges_list.append(torch.from_numpy(edge))
    return torch.stack(edges_list, dim=0)


if __name__ == '__main__':
    img_src = Image.open("_a000001.jpg").convert("RGB")
    edge_src = canny_from_pil(img_src, 20, 50)
    edge_src.show()
    img_tar = Image.open("_b000001.jpg").convert("RGB")
    edge_tar = canny_from_pil(img_tar, 120, 180)
    edge_tar.show()