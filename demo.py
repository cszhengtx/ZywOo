import torch
from PIL import Image
import numpy as np
from torch import nn
import scipy.io as sio
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
import yaml
import random
from model import UNet_DILA
import torch.nn.functional as F
from siren_pytorch import SirenNet, SirenWrapper
from utils import to_coordinates_and_features
import utils
from scipy.ndimage import binary_dilation
import os
import tqdm


net = SirenNet(
            dim_in=2,  # input dimension, ex. 2d coor
            dim_hidden=32,  # hidden dimension
            dim_out=3,  # output dimension, ex. rgb value
            num_layers=6,  # number of layers
            w0_initial=30,
            # different signals may require different omega_0 in the first layer - this is a hyperparameter
            weight_decay=2,
            dropout=True
        ).cuda()
wrapper = SirenWrapper(
            net,
            image_width=512,
            image_height=512
        ).cuda()
optimizer_inr = torch.optim.Adam([
            {'params': wrapper.parameters()}],
            lr=1e-5
        )


img_H = torch.randn(3, 512, 512)
coordinates_H, features_H = to_coordinates_and_features(img_H)
coordinates_H, features_H = coordinates_H.cuda(), features_H.cuda()
img_H = img_H.unsqueeze(0)
MSELoss = torch.nn.MSELoss()
L1Loss = torch.nn.L1Loss()

psnr_total = 0.
ssim_total = 0.
num = 0

def morphological_convolution(mask, structure=None):
    mask_np = mask.cpu().numpy()

    if structure is None:
        structure = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]])

    structure = structure[np.newaxis, np.newaxis, :, :]

    output_np = np.array([
        [[binary_dilation(mask_np[b, c], structure=structure[0, 0]).astype(np.float32)
          for c in range(mask_np.shape[1])]
         for b in range(mask_np.shape[0])]
    ])

    output = torch.from_numpy(output_np.squeeze()).to(mask.device)

    return output


with open('configs/sidd.yaml', 'r') as f:
    config = yaml.safe_load(f)

noisy_orig_np = np.float32(Image.open(r"data\0999_N.png"))
clean_orig_np = np.float32(Image.open(r"data\0999_CL.png"))
img_H_torch = torch.from_numpy(clean_orig_np).permute(2, 0, 1).unsqueeze(0).cuda()
img_L_torch = torch.from_numpy(noisy_orig_np / 255.).permute(2, 0, 1).unsqueeze(0).cuda()
img_orig_torch = img_L_torch

mask_ratio_1 = 0.1

model = UNet_DILA(in_channels=3, out_channels=3, dropout_ratio=0.5).cuda()
model.train()

criteron = nn.L1Loss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['num_iterations'])

for iter in range(config['num_iterations']):
    with torch.no_grad():

        mask_1 = torch.rand(1, img_L_torch.shape[1], img_L_torch.shape[2], img_L_torch.shape[3])
        mask_1 = (mask_1 < mask_ratio_1).float().cuda()

        mask_2 = morphological_convolution(mask_1)

        mask_input = mask_1 * img_L_torch
        target = (1-mask_2) * img_L_torch


    if iter % 5 == 0:
        coordinates, features = to_coordinates_and_features(mask_input.squeeze(0).detach())
        coordinates, features = coordinates.cuda(), features.cuda()
        input_coordinates, input_features, indices = utils.input_shuffle(coordinates_H, features_H)
        out_inr = wrapper(input_coordinates)

        out_inr = out_inr[indices].reshape(img_H.shape[1], img_H.shape[2],
                                           img_H.shape[3]).float().unsqueeze(0)
        out_inr_all = F.pixel_unshuffle(out_inr, 2)
        out_inr_all = out_inr_all.reshape(-1, img_L_torch.shape[1], img_L_torch.shape[2], img_L_torch.shape[3])
        out_index = random.randint(0, 3)
        out_inr_1 = out_inr_all[0]
        loss_inr = MSELoss(mask_input, mask_1 * out_inr_1)

        out_inr_2 = out_inr_all[out_index]

        out_inr_3 = mask_input + (1 - mask_1) * out_inr_2
        output = model(out_inr_3)
        loss = criteron((1 - mask_2) * output, (1 - mask_2) * img_L_torch)
        loss_out = MSELoss((1 - mask_2) * output.detach(), (1 - mask_2) * out_inr_2)
        loss_total = loss_inr + loss + loss_out
    else:
        out_inr_3 = mask_input + (1 - mask_1) * out_inr_2.detach()
        output = model(out_inr_3)
        loss = criteron((1 - mask_2) * output, (1 - mask_2) * img_L_torch)
        loss_total = loss

    optimizer.zero_grad()
    optimizer_inr.zero_grad()
    loss_total.backward()
    optimizer.step()
    scheduler.step()

avg = 0.
avg_out = img_L_torch
for _ in range(config['num_predictions']):
    with torch.no_grad():
        mask_1 = torch.rand(1, img_L_torch.shape[1], img_L_torch.shape[2], img_L_torch.shape[3])
        mask_1 = (mask_1 < mask_ratio_1).float().cuda()

        mask_2 = morphological_convolution(mask_1)

        mask_input = mask_1 * img_L_torch

        input_coordinates, input_features, indices = utils.input_shuffle(coordinates_H, features_H)
        out_inr = wrapper(input_coordinates)
        out_inr = out_inr[indices].reshape(img_H.shape[1], img_H.shape[2],
                                           img_H.shape[3]).float().unsqueeze(0)
        out_inr_all = F.pixel_unshuffle(out_inr, 2)
        out_index = random.randint(0, 3)
        out_inr_all = out_inr_all.reshape(-1, img_L_torch.shape[1], img_L_torch.shape[2], img_L_torch.shape[3])
        out_inr = out_inr_all[out_index]
        out_inr = mask_input + (1 - mask_1) * out_inr
        output = model(out_inr)
        to_img = output.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
        avg += to_img


# avg = avg_out.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()*float(config['num_predictions'])

pred_img = np.clip((avg / float(config['num_predictions'])) * 255., 0., 255.)
image = Image.fromarray(pred_img.astype(np.uint8))

psnr = peak_signal_noise_ratio(np.clip((avg / float(config['num_predictions'])) * 255., 0., 255.),
                               np.array(clean_orig_np), data_range=255)
ssim1 = ssim(np.clip((avg / float(config['num_predictions'])) * 255., 0., 255.), np.array(clean_orig_np),
             channel_axis=2, data_range=255, multichannel=True)
file_name = f'result/_{psnr}_{ssim1}.png'
image.save(file_name)
print(psnr, ssim1)