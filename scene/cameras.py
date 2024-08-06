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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal
from utils.pose_estimation_utils import *

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=torch.tensor([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", gt_mask=None, gt_depth=None, pose_adjustment=True, test=False
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = torch.from_numpy(R).T.cuda()
        self.T = torch.from_numpy(T).cuda()
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.mask = gt_mask.to(self.data_device) if gt_mask is not None else None
        self.depth = gt_depth.to(self.data_device) if gt_depth is not None else None
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        # self.zfar = 20.0
        self.znear = 0.1

        self.trans = trans
        self.scale = scale
        
        #  pose adjustment 
        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device='cuda')
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device='cuda')
        )

        self.pose_adjustment = pose_adjustment
        # 保存未加噪声的姿态
        self.origin_pose(self.pose_adjustment)
        # 添加测试逻辑
        if test:
            print('add noise')
            self.add_noise()

        # 添加姿态优化logic
        if self.pose_adjustment:
            # print('strat pose adjustment')
            l = [
                {'params': [self.cam_rot_delta], 'lr': 0.001, "name": "cam_rot_delta"},
                {'params': [self.cam_trans_delta], 'lr': 0.001, "name": "cam_trans_delta"},
            ]
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    @property
    def K_transform(self):
        P = torch.eye(3, 3).to(self.data_device)
        P[0][0] = fov2focal(self.FoVx, self.image_width)
        P[1][1] = fov2focal(self.FoVy, self.image_height)
        P[0, 2] = self.image_width / 2
        P[1, 2] = self.image_height / 2

        return torch.hstack([P, torch.zeros(3, 1).to(self.data_device)])
    
    @property
    def camrea_matrix(self):

        return self.K_transform @ self.world_view_transform.T

    @property
    def projection_matrix(self):
        return getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
    
    @property
    def world_view_transform(self):
        return getWorld2View2(self.R, self.T, self.trans, self.scale).transpose(0, 1)

    @property
    def full_proj_transform(self):
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]      

    def update_RT(self, R, t):
        self.R = R.to(device=self.R.device)
        self.T = t.to(device=self.T.device)

    def update_pose(self, converged_threshold=1e-4):
        # R 是c2w， t是w2c
        tau = torch.cat([self.cam_trans_delta, self.cam_rot_delta], axis=0)

        T_w2c = torch.eye(4, device=tau.device)
        T_w2c[0:3, 0:3] = self.R
        T_w2c[0:3, 3] = self.T

        new_w2c = SE3_exp(tau) @ T_w2c

        new_R = new_w2c[0:3, 0:3]
        new_T = new_w2c[0:3, 3]

        converged = tau.norm() < converged_threshold
        self.update_RT(new_R, new_T)

        self.cam_rot_delta.data.fill_(0)
        self.cam_trans_delta.data.fill_(0)
        return converged

    def trans_pose(self, trans, R):
        # R 是c2w， t是w2c
        T_w2c = torch.eye(4, device=self.R.device)
        T_w2c[0:3, 0:3] = self.R
        T_w2c[0:3, 3] = self.T

        T_c2w = torch.inverse(T_w2c)
        
        if isinstance(trans, np.ndarray):
            T_c2w[:3, :3] = torch.from_numpy(R).to(self.R.device) @ T_c2w[:3, :3]
            T_c2w[:, 3] = torch.from_numpy(trans).to(self.R.device) @ T_c2w[:, 3]
        else:
            T_c2w[:3, :3] = R.to(self.R.device) @ T_c2w[:3, :3]
            T_c2w[:, 3] = trans.to(self.R.device) @ T_c2w[:, 3]
        
        new_w2c = torch.inverse(T_c2w)

        new_R = new_w2c[0:3, 0:3]
        new_T = new_w2c[0:3, 3]

        self.update_RT(new_R, new_T)

        # update record not adjustment
        self.origin_pose(self.pose_adjustment)
    
    def origin_pose(self, pose_adjustment):
        if pose_adjustment:
            self.world_view_transform_origin = self.world_view_transform.clone()

    def add_noise(self):
        self.R, self.T = add_delta_to_RT(self.R, self.T)

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, projection_matrix, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        self.projection_matrix = projection_matrix
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

        self.cam_rot_delta = None
        self.cam_trans_delta = None