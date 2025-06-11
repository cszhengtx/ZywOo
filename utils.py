import torch
from torch import nn
from einops import rearrange
import numpy as np
import random
import torch.nn.functional as F

def get_conv(ratio=5, stride=1):
    conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=ratio, stride=stride, padding=0, bias=False)
    conv.weight = torch.nn.Parameter((torch.ones((1,1,ratio, ratio))/(1.0 * ratio * ratio)).cuda())
    return conv

def smooth(noisy, ratio=13, stride=1):
    conv = get_conv(ratio, stride)
    b, c, h, w = noisy.shape
    smoothed = conv(noisy.view(-1, 1, h, w))
    _, _, new_h, new_w = smoothed.shape     
    return smoothed.view(1, c, new_h, new_w).detach()

def reproduce(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def calculate_sliding_std(img, upsampler, kernel_size, stride):   
    slided_mean = smooth(img, kernel_size, stride=stride)
    mean_upsampled = upsampler(slided_mean)
    variance = smooth( (img - mean_upsampled)**2, kernel_size, stride=stride)
    upsampled_variance = upsampler(variance)
    return upsampled_variance.sqrt()


def shuffle_input(img, indices, mask, c, size, k):
    if c == 1:
        img_torch = torch.from_numpy(img).unsqueeze(0)
    else:
        img_torch = torch.from_numpy(img)
    mask_torch = torch.from_numpy(mask).unsqueeze(0).repeat(c, 1, 1)
    img_torch_rearranged = rearrange(img_torch.unsqueeze(1), 'c 1 (h1 h) (w1 w) -> c (h1 w1) h w ', h1=size//k, w1=size//k) # (c H//k W//k k k)
    mask_torch_rearranged = rearrange(mask_torch.unsqueeze(1), 'c 1 (h1 h) (w1 w) -> c (h1 w1) h w ', h1=size//k, w1=size//k)
    img_torch_rearranged = img_torch_rearranged.view(c, -1, k*k)# (c H//k*W//k k*k)
    mask_torch_rearranged, _ = torch.max(mask_torch_rearranged.view(c, -1, k*k), 2, keepdim=True)
    img_torch_reordered = torch.gather(img_torch_rearranged.clone(), dim=-1, index=indices).clone()
    img_torch_reordered_v2 = img_torch_reordered.view(c, -1, k*k)
    # Shuffle the image only at the flat regions (where mask = 0)
    img_torch_final = mask_torch_rearranged * img_torch_rearranged + (1 - mask_torch_rearranged) * img_torch_reordered_v2
    img_torch_final = img_torch_final.view(c, -1, k, k)    
    img_torch_final_v2 = rearrange(img_torch_final, 'c (h1 w1) h w -> c 1 (h1 h) (w1 w) ', h1=size//k, w1=size//k)
    return img_torch_final_v2.squeeze().cpu().numpy() 

def get_shuffling_mask(std_map_torch, threshold=0.5):
    std_map = std_map_torch.cpu().numpy().squeeze()
    normalized = std_map/std_map.max()
    thresholded = np.zeros_like(normalized)
    thresholded[normalized >= threshold] = 1.
    return thresholded


def generate_random_permutation(img_size, c, factor):
    d1, d2, d3 = c, (img_size//factor)*(img_size//factor), factor*factor
    permutaion_indices = torch.argsort(torch.rand(1, d2, d3), dim=-1)
    permutaion_indices = permutaion_indices.repeat(d1, 1, 1)
    reverse_permutation_indices = torch.argsort(permutaion_indices, dim=-1)

    return permutaion_indices, reverse_permutation_indices

# def pixel_shuffle_down_sampling(x:torch.Tensor, f:int, pad:int=0, pad_value:float=0.):
#     '''
#     pixel-shuffle down-sampling (PD) from "When AWGN-denoiser meets real-world noise." (AAAI 2019)
#     Args:
#         x (Tensor) : input tensor
#         f (int) : factor of PD
#         pad (int) : number of pad between each down-sampled images
#         pad_value (float) : padding value
#     Return:
#         pd_x (Tensor) : down-shuffled image tensor with pad or not
#     '''
#     # single image tensor
#     if len(x.shape) == 3:
#         c,w,h = x.shape
#         unshuffled = F.pixel_unshuffle(x, f)
#         if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
#         return unshuffled.view(c,f,f,w//f+2*pad,h//f+2*pad).permute(0,1,3,2,4).reshape(c, w+2*f*pad, h+2*f*pad)
#     # batched image tensor
#     else:
#         b,c,w,h = x.shape
#         unshuffled = F.pixel_unshuffle(x, f)
#         if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
#         return unshuffled.view(b,c,f,f,w//f+2*pad,h//f+2*pad).permute(0,1,2,4,3,5).reshape(b,c,w+2*f*pad, h+2*f*pad)
#
# def pixel_shuffle_up_sampling(x:torch.Tensor, f:int, pad:int=0):
#     '''
#     inverse of pixel-shuffle down-sampling (PD)
#     see more details about PD in pixel_shuffle_down_sampling()
#     Args:
#         x (Tensor) : input tensor
#         f (int) : factor of PD
#         pad (int) : number of pad will be removed
#     '''
#     # single image tensor
#     if len(x.shape) == 3:
#         c,w,h = x.shape
#         before_shuffle = x.view(c,f,w//f,f,h//f).permute(0,1,3,2,4).reshape(c*f*f,w//f,h//f)
#         if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
#         return F.pixel_shuffle(before_shuffle, f)
#     # batched image tensor
#     else:
#         b,c,w,h = x.shape
#         before_shuffle = x.view(b,c,f,w//f,f,h//f).permute(0,1,2,4,3,5).reshape(b,c*f*f,w//f,h//f)
#         if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
#         return F.pixel_shuffle(before_shuffle, f)


def pixel_shuffle_down_sampling_random(x:torch.Tensor, f: int, pad:int=0, pad_value:float=0.):
    '''
    pixel-shuffle down-sampling (PD) from "When AWGN-denoiser meets real-world noise." (AAAI 2019)
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad between each down-sampled images
        pad_value (float) : padding value
    Return:
        pd_x (Tensor) : down-shuffled image tensor with pad or not
    '''
    # single image tensor
    if len(x.shape) == 3:
        c,w,h = x.shape
        x = F.pixel_unshuffle(x, f)
        if pad != 0: x = F.pad(x, (pad, pad, pad, pad), value=pad_value)

        channel_indices = torch.randperm(c)
        # 对输入张量的channel维度进行shuffle
        x = x[channel_indices, :, :]

        return x.view(c,f,f,w//f+2*pad,h//f+2*pad).permute(0,1,3,2,4).reshape(c, w+2*f*pad, h+2*f*pad)
    # batched image tensor
    else:
        b,c,w,h = x.shape
        x = F.pixel_unshuffle(x, f)
        if pad != 0: x = F.pad(x, (pad, pad, pad, pad), value=pad_value)
        # channel_indices = torch.randperm(x.shape[1])
        # 对输入张量的channel维度进行shuffle
        # unshuffled = unshuffled[:, channel_indices, :, :]
        channel_index = []
        sub_img = []
        for i in range(b):
            sub_sub_img = []
            for j in range(f ** 2):
                sub_sub_img.append(x[i, j::f ** 2, :, :])
            channel_indices = torch.randperm(f ** 2)
            channel_index.append(channel_indices)
            sub_sub_img = torch.stack(sub_sub_img,dim=0)[channel_indices, :, :, :]
            sub_img.append(sub_sub_img.reshape(-1, sub_sub_img.shape[-2], sub_sub_img.shape[-1]))
        x = torch.stack(sub_img, dim=0)
        # 对输入张量的channel维度进行shuffle
        return x.view(b,c,f,f,w//f+2*pad,h//f+2*pad).permute(0,1,2,4,3,5).reshape(b,c,w+2*f*pad, h+2*f*pad), channel_index


def pixel_shuffle_up_sampling_random(x:torch.Tensor, channel_index, f:int, pad:int=0):
    '''
    inverse of pixel-shuffle down-sampling (PD)
    see more details about PD in pixel_shuffle_down_sampling()
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad will be removed
        channel_indices :
    '''
    # single image tensor
    if len(x.shape) == 3:
        c,w,h = x.shape
        before_shuffle = x.view(c,f,w//f,f,h//f).permute(0,1,3,2,4).reshape(c*f*f,w//f,h//f)
        if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)
    # batched image tensor
    else:
        b,c,w,h = x.shape
        before_shuffle = x.view(b,c,f,w//f,f,h//f).permute(0,1,2,4,3,5).reshape(b,c*f*f,w//f,h//f)
        sub_img = []
        for i in range(b):
            sub_sub_img = []
            channel_indices = torch.argsort(channel_index[i])
            for j in range(f ** 2):
                sub_sub_img.append(before_shuffle[i, j::f ** 2, :, :])
            sub_sub_img = torch.stack(sub_sub_img, dim=0)[channel_indices, :, :, :]
            sub_img.append(sub_sub_img.reshape(-1, sub_sub_img.shape[-2], sub_sub_img.shape[-1]))
        before_shuffle = torch.stack(sub_img, dim=0)
            # unshuffled_indices = torch.argsort(channel_indices)
        # before_shuffle = before_shuffle[:, unshuffled_indices, :, :]

        if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)

def pixel_shuffle_down_sampling(x:torch.Tensor, f:int, pad:int=0, pad_value:float=0.):
    '''
    pixel-shuffle down-sampling (PD) from "When AWGN-denoiser meets real-world noise." (AAAI 2019)
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad between each down-sampled images
        pad_value (float) : padding value
    Return:
        pd_x (Tensor) : down-shuffled image tensor with pad or not
    '''
    # single image tensor
    if len(x.shape) == 3:
        c,w,h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(c,f,f,w//f+2*pad,h//f+2*pad).permute(0,1,3,2,4).reshape(c, w+2*f*pad, h+2*f*pad)
    # batched image tensor
    else:
        b,c,w,h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(b,c,f,f,w//f+2*pad,h//f+2*pad).permute(0,1,2,4,3,5).reshape(b,c,w+2*f*pad, h+2*f*pad)

def pixel_shuffle_up_sampling(x:torch.Tensor, f:int, pad:int=0):
    '''
    inverse of pixel-shuffle down-sampling (PD)
    see more details about PD in pixel_shuffle_down_sampling()
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad will be removed
    '''
    # single image tensor
    if len(x.shape) == 3:
        c,w,h = x.shape
        before_shuffle = x.view(c,f,w//f,f,h//f).permute(0,1,3,2,4).reshape(c*f*f,w//f,h//f)
        if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)
    # batched image tensor
    else:
        b,c,w,h = x.shape
        before_shuffle = x.view(b,c,f,w//f,f,h//f).permute(0,1,2,4,3,5).reshape(b,c*f*f,w//f,h//f)
        if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)


from torchvision import transforms
from tqdm import tqdm
import cv2
import numpy as np
from scipy.signal import convolve2d
import math
import torch
import torch.fft as fft
import torch.nn as nn
from torch.nn import functional as F
from skimage import img_as_float
import time

def im2patch(im, pch_size, stride=1):
    '''
    Transform image to patches.
    Input:
        im: 3 x H x W or 1 X H x W image, numpy format
        pch_size: (int, int) tuple or integer
        stride: (int, int) tuple or integer
    '''
    if isinstance(pch_size, tuple):
        pch_H, pch_W = pch_size
    elif isinstance(pch_size, int):
        pch_H = pch_W = pch_size
    else:
        sys.exit('The input of pch_size must be a integer or a int tuple!')

    if isinstance(stride, tuple):
        stride_H, stride_W = stride
    elif isinstance(stride, int):
        stride_H = stride_W = stride
    else:
        sys.exit('The input of stride must be a integer or a int tuple!')


    C, H, W = im.shape
    num_H = len(range(0, H-pch_H+1, stride_H))
    num_W = len(range(0, W-pch_W+1, stride_W))
    num_pch = num_H * num_W
    pch = np.zeros((C, pch_H*pch_W, num_pch), dtype=im.dtype)
    kk = 0
    for ii in range(pch_H):
        for jj in range(pch_W):
            temp = im[:, ii:H-pch_H+ii+1:stride_H, jj:W-pch_W+jj+1:stride_W]
            pch[:, kk, :] = temp.reshape((C, num_pch))
            kk += 1

    return pch.reshape((C, pch_H, pch_W, num_pch))

def noise_estimate(im, pch_size=8):
    '''
    Implement of noise level estimation of the following paper:
    Chen G , Zhu F , Heng P A . An Efficient Statistical Method for Image Noise Level Estimation[C]// 2015 IEEE International Conference
    on Computer Vision (ICCV). IEEE Computer Society, 2015.
    Input:
        im: the noise image, H x W x 3 or H x W numpy tensor, range [0,1]
        pch_size: patch_size
    Output:
        noise_level: the estimated noise level
    '''
    im = img_as_float(im)
    if im.ndim == 3:
        im = im.transpose((2, 0, 1))
    else:
        im = np.expand_dims(im, axis=0)

    # image to patch
    pch = im2patch(im, pch_size, 3)  # C x pch_size x pch_size x num_pch tensor
    num_pch = pch.shape[3]
    pch = pch.reshape((-1, num_pch))  # d x num_pch matrix
    d = pch.shape[0]

    mu = pch.mean(axis=1, keepdims=True)  # d x 1
    X = pch - mu
    sigma_X = np.matmul(X, X.transpose()) / num_pch
    sig_value, _ = np.linalg.eigh(sigma_X)
    sig_value.sort()

    for ii in range(-1, -d-1, -1):
        tau = np.mean(sig_value[:ii])
        if np.sum(sig_value[:ii]>tau) == np.sum(sig_value[:ii] < tau):
            return np.sqrt(tau)

MSELoss = torch.nn.MSELoss()


def log_to_txt(filename, content):
    with open(filename, 'a') as file:
        file.write(content + '\n')


def add_gaussian_noise(image, variance):
    noise = torch.randn_like(image) * (variance/100)
    noisy_image = image + noise
    return noisy_image


def hf_loss(output):
    loss = -torch.sqrt(torch.var(output))
    return loss


def to_coordinates(img):
    """Converts an image to a set of coordinates and features.

    Args:
        img (torch.Tensor): Shape (channels, height, width).
        gama (int) : The coordinate multiples of superresolution
    """
    # Coordinates are indices of all non zero locations of a tensor of ones of
    # same shape as spatial dimensions of image
    coordinates = torch.ones(img.shape[1:]).nonzero(as_tuple=False).float() #
    # Normalize coordinates to lie in [-.5, .5]
    coordinates = coordinates / (img.shape[1] - 1) - 0.5
    # Convert to range [-1, 1]
    coordinates *= 2
    # Convert image to a tensor of features of shape (num_points, channels)
    return coordinates

def generate_fourier_feature_maps(coordinates, size=8, only_cosine=False):
    freqs = 2 ** (size / (size - 1)) ** torch.linspace(0., size - 1, steps=size)
    vp = freqs * coordinates.unsqueeze(-1)
    if only_cosine:
        vp_cat = torch.cat((torch.cos(vp),), dim=-1)
    else:
        vp_cat = torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)
    vp_cat = vp_cat.flatten(1)
    return vp_cat


def random_fourier_features(x, num_features, sigma=1.0):
    """
    随机傅里叶特征近似
    :param x: 输入数据，形状为 [batch_size, input_dim]
    :param num_features: 特征维度，即生成的傅里叶特征数量
    :param sigma: 高斯核的参数，控制特征的分布
    :return: 近似特征，形状为 [batch_size, num_features]
    """
    input_dim = x.shape[1]

    # 随机生成傅里叶基函数的参数
    omega = torch.randn(input_dim, num_features) * (2.0 * np.pi)
    b = torch.rand(1, num_features) * (2.0 * np.pi)

    # 计算随机傅里叶特征
    projection = torch.cos(torch.mm(x, omega) + b)

    # 缩放以适应高斯核
    return projection / np.sqrt(num_features) * sigma


def add_noise_to_features(img, variance=10):
    noise_img = add_gaussian_noise(img, variance)
    features = noise_img.reshape(img.shape[0], -1).T

    return features


def add_blur_to_features(img, kernel_size, sigma):
    blur_img = gaussian_low_pass_filter(img, kernel_size, sigma)
    blur_img = transforms.ToTensor()(blur_img).float()
    features = blur_img.reshape(img.shape[0], -1).T
    return features

def create_gaussian_kernel(kernel_size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1/ (2*np.pi*sigma**2)) * np.exp(- ((x - (kernel_size-1)/2)**2 + (y - (kernel_size-1)/2)**2) / (2*sigma**2)),
        (kernel_size, kernel_size)
    )
    kernel /= np.sum(kernel)
    return kernel


def gaussian_low_pass_filter(image, kernel_size, sigma):
    kernel = create_gaussian_kernel(kernel_size, sigma)
    filtered_image = np.zeros_like(image)
    for channel in range(3):
        filtered_image[:, :, channel] = convolve2d(image[:, :, channel], kernel, mode='same', boundary='wrap')
    return filtered_image


def tensor_to_numpy(img_tensor):
    img_array = img_tensor.cpu().numpy()

    # 将像素值映射到[0, 255]
    img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min())) * 255

    # 将浮点数转换为整数
    img_array = img_array.astype(np.uint8)

    return img_array


def save_image_tensor2cv2(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为cv2格式
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    # 去掉批次维度
    input_tensor = input_tensor.squeeze()
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    # RGB转BRG
    # input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, input_tensor)

def to_coordinates_and_features(img):
    """Converts an image to a set of coordinates and features.

    Args:
        img (torch.Tensor): Shape (channels, height, width).
        gama (int) : The coordinate multiples of superresolution
    """
    # Coordinates are indices of all non zero locations of a tensor of ones of
    # same shape as spatial dimensions of image
    coordinates = torch.ones(img.shape[1:]).nonzero(as_tuple=False).float() #
    # Normalize coordinates to lie in [-.5, .5]
    coordinates = coordinates / (img.shape[1] - 1) - 0.5
    # Convert to range [-1, 1]
    coordinates *= 2
    # Convert image to a tensor of features of shape (num_points, channels)
    features = img.reshape(img.shape[0], -1).T
    return coordinates, features


def input_shuffle(coordinates, features):
    indices = torch.randperm(coordinates.size(0))

    # 使用索引顺序对coordinates和features进行shuffle
    shuffled_coordinates = coordinates[indices]
    shuffled_features = features[indices]

    return shuffled_coordinates, shuffled_features, torch.argsort(indices)

L1Loss = nn.L1Loss()
loss_fn = nn.MSELoss()


def compare_psnr(original, noisy):
    mse = F.mse_loss(original, noisy)
    psnr = 20 * torch.log10(torch.max(original) / torch.sqrt(mse))
    return psnr


def get_noisy_image(img_np, sigma):
    # sigma = sigma/255.
    img_noise_np = np.clip(img_np + np.random.normal(scale=sigma, size=img_np.shape), 0, 1).astype(np.float32)

    return img_noise_np