from PIL import Image
import cv2
import numpy as np
import torch

from .view_base import BaseView

ORTH_TRANSFORM = np.array([[-0.1466152, 0.93333838, -0.32769415],
                           [-0.96469466, -0.20818096, -0.16132233],
                           [-0.21878801, 0.29247249, 0.93090905]])

ORTH_TRANSFORM_INV = np.linalg.inv(ORTH_TRANSFORM)

def simulate_colorblindness(image, transform=None):
    M = transform if transform else ORTH_TRANSFORM
    converted_image = cv2.transform(image, M)

    return converted_image

def inverse_colorblindness(image, transform=None):
    M_inv = transform if transform else ORTH_TRANSFORM_INV
    reverted_image = cv2.transform(image, M_inv)

    return reverted_image

class ColorblindView(BaseView):
    '''
    ColorblindView: transform the RGB image into a colorblind-friendly color space
    '''
    def __init__(self):
        super().__init__()
        self.colorblind_type = "generic"

    def __str__(self):
        return f"ColorblindView: (type={self.colorblind_type})"

    def view(self, im):
        dev, im_dtype = im.device, im.dtype
        im = im.to(torch.float32)
        im = im.permute(1, 2, 0)
        im = np.array(im.cpu())
        im_res = im
        im_res[:, :, :3] = simulate_colorblindness(im[:, :, :3])
        im_res = torch.tensor(im_res, dtype=im_dtype).to(dev)
        im_res = im_res.permute(2, 0, 1)
        return im_res
    
    def inverse_view(self, noise):
        dev, noise_dtype = noise.device, noise.dtype
        noise = noise.to(torch.float32)
        noise = noise.permute(1, 2, 0) 
        noise = np.array(noise.cpu())
        noise_res = noise
        noise_res[:, :, :3] = inverse_colorblindness(noise[:, :, :3])
        noise_res = torch.tensor(noise_res, dtype=noise_dtype).to(dev)
        noise_res = noise_res.permute(2, 0, 1)
        return noise_res