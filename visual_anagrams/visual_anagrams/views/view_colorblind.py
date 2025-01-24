from PIL import Image
import cv2
import numpy as np
import torch

from .view_base import BaseView

PROTANOPIA_TRANSFORM = np.array([[0.567, 0.433, 0.0],
                                 [0.558, 0.442, 0.0],
                                 [0.0,   0.242, 0.758]])

PROTANOPIA_TRANSFORM_INV = np.linalg.inv(PROTANOPIA_TRANSFORM)

DEUTERANOPIA_TRANSFORM = np.array([[-0.79031177, -0.54297324, -0.28387913],
                                   [-0.1458037, -0.28334704, 0.94786905],
                                   [-0.59510384,  0.7905027,  0.14476499]])

DEUTERANOPIA_TRANSFORM_INV = np.linalg.inv(DEUTERANOPIA_TRANSFORM)

TRITANOPIA_TRANSFORM = np.array([[0.95,  0.05,  0.0],
                                 [0.0,   0.433, 0.567],
                                 [0.0,   0.475, 0.525]])

TRITANOPIA_TRANSFORM_INV = np.linalg.inv(TRITANOPIA_TRANSFORM)

# PROTANOPIA_TRANSFORM = torch.tensor([[0.567, 0.433, 0.0],
#                                      [0.558, 0.442, 0.0],
#                                      [0.0,   0.242, 0.758]], dtype=torch.float16)

# PROTANOPIA_TRANSFORM_INV = torch.inverse(PROTANOPIA_TRANSFORM)

# DEUTERANOPIA_TRANSFORM = torch.tensor([[0.625, 0.375, 0.0],
#                                    [0.7,   0.3,   0.0],
#                                    [0.0,   0.3,   0.7]], dtype=torch.float16)

# DEUTERANOPIA_TRANSFORM_INV = torch.inverse(DEUTERANOPIA_TRANSFORM)

# TRITANOPIA_TRANSFORM = torch.tensor([[0.95,  0.05,  0.0],
#                                      [0.0,   0.433, 0.567],
#                                      [0.0,   0.475, 0.525]], dtype=torch.float16)

# TRITANOPIA_TRANSFORM_INV = torch.inverse(TRITANOPIA_TRANSFORM)

def simulate_colorblindness(image, blindness_type):
    if blindness_type == 'protanopia':
        M = PROTANOPIA_TRANSFORM
    elif blindness_type == 'deuteranopia':
        M = DEUTERANOPIA_TRANSFORM
    elif blindness_type == 'tritanopia':
        M = TRITANOPIA_TRANSFORM
    else:
        raise ValueError("Invalid colorblindness type.")

    blind_image = cv2.transform(image, M)

    return blind_image

def inverse_colorblindness(image, blindness_type):
    if blindness_type == 'protanopia':
        M_inv = None
    elif blindness_type == 'deuteranopia':
        M_inv = DEUTERANOPIA_TRANSFORM_INV
    elif blindness_type == 'tritanopia':
        M_inv = None
    else:
        raise ValueError("Invalid colorblindness type.")

    reverted_image = cv2.transform(image, M_inv)

    return reverted_image

# def simulate_colorblindness(image, blindness_type):
#     if blindness_type == 'protanopia':
#         M = PROTANOPIA_TRANSFORM
#     elif blindness_type == 'deuteranopia':
#         M = DEUTERANOPIA_TRANSFORM
#     elif blindness_type == 'tritanopia':
#         M = TRITANOPIA_TRANSFORM
#     else:
#         raise ValueError("Invalid colorblindness type.")

#     M = M.to(image.device)

#     image = image.permute(1, 2, 0)
#     blind_image = torch.tensordot(image, M, dims=1).permute(2, 0, 1)

#     return blind_image

# def inverse_colorblindness(image, blindness_type):
#     if blindness_type == 'protanopia':
#         M_inv = PROTANOPIA_TRANSFORM_INV
#     elif blindness_type == 'deuteranopia':
#         M_inv = DEUTERANOPIA_TRANSFORM_INV
#     elif blindness_type == 'tritanopia':
#         M_inv = TRITANOPIA_TRANSFORM_INV
#     else:
#         raise ValueError("Invalid colorblindness type.")

#     M_inv = M_inv.to(image.device)

#     image = image.permute(1, 2, 0)
#     reverted_image = torch.tensordot(image, M_inv, dims=1).permute(2, 0, 1)

#     return reverted_image

class ColorblindView(BaseView):
    '''
    Converts RGB image to colorblind space
    '''
    def __init__(self):
        super().__init__()

    def view(self, im):
        dev, im_dtype = im.device, im.dtype
        im = im.to(torch.float32)
        im = im.permute(1, 2, 0)
        im = np.array(im.cpu())
        im_res = im
        im_res[:, :, :3] = simulate_colorblindness(im[:, :, :3], "deuteranopia")
        im_res = torch.tensor(im_res, dtype=im_dtype).to(dev)
        im_res = im_res.permute(2, 0, 1)
        return im_res
    
    def inverse_view(self, noise):
        dev, noise_dtype = noise.device, noise.dtype
        noise = noise.to(torch.float32)
        noise = noise.permute(1, 2, 0) 
        noise = np.array(noise.cpu())
        noise_res = noise
        noise_res[:, :, :3] = inverse_colorblindness(noise[:, :, :3], "deuteranopia")
        noise_res = torch.tensor(noise_res, dtype=noise_dtype).to(dev)
        noise_res = noise_res.permute(2, 0, 1)
        return noise_res
    
    def make_frame(self, im, t):
        im_size = im.size[0]
        frame_size = int(im_size * 1.5)

        im = self.view(im)
        im = Image.fromarray((np.array(im) * 255.).astype(np.uint8))

        frame = Image.new('RGB', (frame_size, frame_size), (255, 255, 255))
        frame.paste(im, ((frame_size - im_size) // 2, (frame_size - im_size) // 2))

        return frame