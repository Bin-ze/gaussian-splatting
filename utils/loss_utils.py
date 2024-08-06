#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def depth_loss_dpt(pred_depth, gt_depth, weight=None):
    """
    :param pred_depth:  (H, W)
    :param gt_depth:    (H, W)
    :param weight:      (H, W)
    :return:            scalar
    """
    t_pred = torch.median(pred_depth)
    s_pred = torch.mean(torch.abs(pred_depth - t_pred))

    t_gt = torch.median(gt_depth)
    s_gt = torch.mean(torch.abs(gt_depth - t_gt))

    pred_depth_n = (pred_depth - t_pred) / s_pred
    gt_depth_n = (gt_depth - t_gt) / s_gt

    if weight is not None:
        loss = F.mse_loss(pred_depth_n, gt_depth_n, reduction='none')
        loss = loss * weight
        loss = loss.sum() / (weight.sum() + 1e-8)
    else:

        depth_error = (pred_depth_n - gt_depth_n) ** 2
        depth_error[depth_error > torch.quantile(depth_error, 0.8)] = 0
        loss = depth_error.mean()
        # loss = F.mse_loss(pred_depth_n, gt_depth_n)

    return loss

def compute_depth_loss(dyn_depth, gt_depth, lambda_depth):
    """
    参考https://github1s.com/facebookresearch/localrf/blob/main/localTensoRF/utils/utils.py中的实现
    Inputs:
    dyn_depth: nerf(3d gs)渲染视差
    gt_depth: 单目深度估计视差
    lambda_depth: 损失加权系数
    Outputs:
        depth_loss: 该损失函数可以用于缓解渲染深度与监督深度之间存在的尺度不一致的问题
    """
    dyn_depth = dyn_depth.view(1, -1)
    gt_depth = gt_depth.view(1, -1)
    t_d = torch.median(dyn_depth, dim=-1, keepdim=True).values
    s_d = torch.mean(torch.abs(dyn_depth - t_d), dim=-1, keepdim=True)
    dyn_depth_norm = (dyn_depth - t_d) / s_d

    t_gt = torch.median(gt_depth, dim=-1, keepdim=True).values
    s_gt = torch.mean(torch.abs(gt_depth - t_gt), dim=-1, keepdim=True)
    gt_depth_norm = (gt_depth - t_gt) / s_gt
    depth_loss_arr = (dyn_depth_norm - gt_depth_norm) ** 2
    depth_loss_arr[depth_loss_arr > torch.quantile(depth_loss_arr, 0.8, dim=1)[..., None]] = 0
    depth_loss = (depth_loss_arr).mean() * lambda_depth
    return  depth_loss

def compute_rank_loss(dyn_depth, gt_depth, lambda_depth, sample_nums=1000):
    """
    参考论文Sparse-Nerf：https://sparsenerf.github.io/的描述进行实现
    Inputs:
        dyn_depth: nerf(3d gs)渲染视差
        gt_depth: 单目深度估计视差
        lambda_depth: 损失加权系数
        sample_nums: 单张图像上用于监督的采样点个数
    Outputs:
        rank_loss:描述了由nerf等算法渲染的深度与单目深度之间的关系，如果两者有相同的前后关系，则损失较低，否则，损失较高
    """
    pred_depth = dyn_depth.view(1, -1) / dyn_depth.max()
    gt_depth = gt_depth.view(1, -1) / gt_depth.max()
    # 随机取1000个样本
    sample = torch.randint(0, pred_depth.shape[1], (sample_nums,))
    # 采样
    pred_depth = pred_depth[:, sample]
    gt_depth = gt_depth[:, sample]
    # 直接满足前后关系
    mask_rank = torch.where(gt_depth.unsqueeze(-1) - gt_depth.unsqueeze(1) > 1e-4, 1, 0).type(torch.bool)
    rank_loss = (pred_depth.unsqueeze(1) - pred_depth.unsqueeze(-1))[mask_rank].clamp(0).mean() * lambda_depth

    return rank_loss


def compute_scale_regularization_loss(gaussians, visibility_filter, lambda_1=1e-3, lambda_2=1e-4):
    """
    为了缓解场景中存在的线状高斯，设计了一个正则损失，去约束高斯的尺度不要朝着无限大的方向优化
    Inputs:
        gaussians: 3d gs场景表示
        visibility_filter: 单张图像看到的范围
        lambda_*: 损失加权
    Outputs:
        scale_regularization_loss: 当场景gs被优化为各项同性时,损失函数最小为0
    """
    scale_ = gaussians.get_scaling[visibility_filter]
    scale_1, scale_2, scale_3 = scale_[:, 0] / scale_[:, 1], scale_[:, 1] / scale_[:, 2], scale_[:, 0] / scale_[:, 2]
    # 将其都转换为大于1的数
    scale_1 = torch.where(scale_1 < 1, 1 / scale_1, scale_1)
    scale_2 = torch.where(scale_2 < 1, 1 / scale_2, scale_2)
    scale_3 = torch.where(scale_3 < 1, 1 / scale_3, scale_3)
    # 对于每个比值，进行正则化
    scale_regularization_loss = lambda_1 * torch.log(((scale_1).mean() + (scale_2).mean() + (scale_3).mean() - 3) / 3 + 1)
    # + lambda_2 * ((scale_1).max() + (scale_2).max() + (scale_3).max() - 3) / 3 # 第一项约束全部， 第二项约束离群
    return scale_regularization_loss
    
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

