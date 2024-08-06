#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   pose_estimation_utils.py
@Time    :   2024/03/21 16:45:53
@Author  :   Bin-ze 
@Version :   1.0
@Desc    :  Description of the script or module goes here. 
储存来一些用于姿态优化的一些辅助函数以及测试函数
'''

import json
import torch
import numpy as np
from typing import Tuple

from .graphics_utils import fov2focal


def rt2mat(R, T):
    mat = np.eye(4)
    mat[0:3, 0:3] = R
    mat[0:3, 3] = T
    return mat


def skew_sym_mat(x):
    device = x.device
    dtype = x.dtype
    ssm = torch.zeros(3, 3, device=device, dtype=dtype)
    ssm[0, 1] = -x[2]
    ssm[0, 2] = x[1]
    ssm[1, 0] = x[2]
    ssm[1, 2] = -x[0]
    ssm[2, 0] = -x[1]
    ssm[2, 1] = x[0]
    return ssm


def SO3_exp(theta):
    device = theta.device
    dtype = theta.dtype

    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    I = torch.eye(3, device=device, dtype=dtype)
    if angle < 1e-5:
        return I + W + 0.5 * W2
    else:
        return (
            I
            + (torch.sin(angle) / angle) * W
            + ((1 - torch.cos(angle)) / (angle**2)) * W2
        )


def V(theta):
    dtype = theta.dtype
    device = theta.device
    I = torch.eye(3, device=device, dtype=dtype)
    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    if angle < 1e-5:
        V = I + 0.5 * W + (1.0 / 6.0) * W2
    else:
        V = (
            I
            + W * ((1.0 - torch.cos(angle)) / (angle**2))
            + W2 * ((angle - torch.sin(angle)) / (angle**3))
        )
    return V


def SE3_exp(tau):
    dtype = tau.dtype
    device = tau.device

    rho = tau[:3]
    theta = tau[3:]
    R = SO3_exp(theta)
    t = V(theta) @ rho

    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def quaternion_to_rotation_matrix_torch(q):
    """
    Convert a quaternion into a full three-dimensional rotation matrix.

    Input:
    :param q: A tensor of size (B, 4), where B is batch size and quaternion is in format (x, y, z, w).

    Output:
    :return: A tensor of size (B, 3, 3), where B is batch size.
    """
    # Ensure quaternion has four components
    assert q.shape[-1] == 4, "Input quaternion should have 4 components!"

    w, x, y, z = q.unbind(-1)

    # Compute quaternion norms
    q_norm = torch.norm(q, dim=-1, keepdim=True)
    # Normalize input quaternions
    q = q / q_norm

    # Compute the quaternion outer product
    q_outer = torch.einsum('...i,...j->...ij', q, q)

    # Compute rotation matrix
    rot_matrix = torch.empty(
        (*q.shape[:-1], 3, 3), dtype=q.dtype, device=q.device)
    rot_matrix[..., 0, 0] = 1 - 2 * (y**2 + z**2)
    rot_matrix[..., 0, 1] = 2 * (x*y - z*w)
    rot_matrix[..., 0, 2] = 2 * (x*z + y*w)
    rot_matrix[..., 1, 0] = 2 * (x*y + z*w)
    rot_matrix[..., 1, 1] = 1 - 2 * (x**2 + z**2)
    rot_matrix[..., 1, 2] = 2 * (y*z - x*w)
    rot_matrix[..., 2, 0] = 2 * (x*z - y*w)
    rot_matrix[..., 2, 1] = 2 * (y*z + x*w)
    rot_matrix[..., 2, 2] = 1 - 2 * (x**2 + y**2)

    return rot_matrix


def rotation_matrix_to_quaternion_torch(
    R: torch.Tensor # (batch_size, 3, 3)
)->torch.Tensor:
    q = torch.zeros(R.shape[0], 4, device=R.device, dtype=R.dtype) # (batch_size, 4) x, y, z, w
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    q0_mask = trace > 0
    q1_mask = (R[..., 0, 0] > R[..., 1, 1]) & (R[..., 0, 0] > R[..., 2, 2]) & ~q0_mask
    q2_mask = (R[..., 1, 1] > R[..., 2, 2]) & ~q0_mask & ~q1_mask
    q3_mask = ~q0_mask & ~q1_mask & ~q2_mask
    if q0_mask.any():
        R_for_q0 = R[q0_mask]
        S_for_q0 = 0.5 / torch.sqrt(1 + trace[q0_mask])
        q[q0_mask, 3] = 0.25 / S_for_q0
        q[q0_mask, 0] = (R_for_q0[..., 2, 1] - R_for_q0[..., 1, 2]) * S_for_q0
        q[q0_mask, 1] = (R_for_q0[..., 0, 2] - R_for_q0[..., 2, 0]) * S_for_q0
        q[q0_mask, 2] = (R_for_q0[..., 1, 0] - R_for_q0[..., 0, 1]) * S_for_q0
    
    if q1_mask.any():
        R_for_q1 = R[q1_mask]
        S_for_q1 = 2.0 * torch.sqrt(1 + R_for_q1[..., 0, 0] - R_for_q1[..., 1, 1] - R_for_q1[..., 2, 2])
        q[q1_mask, 0] = 0.25 * S_for_q1
        q[q1_mask, 1] = (R_for_q1[..., 0, 1] + R_for_q1[..., 1, 0]) / S_for_q1
        q[q1_mask, 2] = (R_for_q1[..., 0, 2] + R_for_q1[..., 2, 0]) / S_for_q1
        q[q1_mask, 3] = (R_for_q1[..., 2, 1] - R_for_q1[..., 1, 2]) / S_for_q1
    
    if q2_mask.any():
        R_for_q2 = R[q2_mask]
        S_for_q2 = 2.0 * torch.sqrt(1 + R_for_q2[..., 1, 1] - R_for_q2[..., 0, 0] - R_for_q2[..., 2, 2])
        q[q2_mask, 0] = (R_for_q2[..., 0, 1] + R_for_q2[..., 1, 0]) / S_for_q2
        q[q2_mask, 1] = 0.25 * S_for_q2
        q[q2_mask, 2] = (R_for_q2[..., 1, 2] + R_for_q2[..., 2, 1]) / S_for_q2
        q[q2_mask, 3] = (R_for_q2[..., 0, 2] - R_for_q2[..., 2, 0]) / S_for_q2
    
    if q3_mask.any():
        R_for_q3 = R[q3_mask]
        S_for_q3 = 2.0 * torch.sqrt(1 + R_for_q3[..., 2, 2] - R_for_q3[..., 0, 0] - R_for_q3[..., 1, 1])
        q[q3_mask, 0] = (R_for_q3[..., 0, 2] + R_for_q3[..., 2, 0]) / S_for_q3
        q[q3_mask, 1] = (R_for_q3[..., 1, 2] + R_for_q3[..., 2, 1]) / S_for_q3
        q[q3_mask, 2] = 0.25 * S_for_q3
        q[q3_mask, 3] = (R_for_q3[..., 1, 0] - R_for_q3[..., 0, 1]) / S_for_q3
    # xyzw -> wxyz
    wxyz = torch.zeros(R.shape[0], 4, device=R.device, dtype=R.dtype)
    wxyz[0, 0] = q[0, -1]
    wxyz[0, 1:] = q[0, :-1] 
    return wxyz


def se3_to_quaternion_and_translation_torch(
    transform: torch.Tensor, # (batch_size, 4, 4)
)->Tuple[torch.Tensor, torch.Tensor]:
    R = transform[..., :3, :3]
    t = transform[..., :3, 3]
    q = rotation_matrix_to_quaternion_torch(R)
    return q, t


def world_view_transform_to_quaternion_and_translation_torch(
    transform: torch.Tensor, # (batch_size, 4, 4)
)->Tuple[torch.Tensor, torch.Tensor]:
    R = transform[..., :3, :3]
    t = transform[..., :3, 3]
    q = rotation_matrix_to_quaternion_torch(R)
    return q, t

def save_pose(viewpoint_stack, save_path='./', r = 2, stuff=None):
    print('save pose')
    if r == -1:
        r = 1
    Ks, c2ws_org, c2ws_adjust, file_names = [], [], [], []

    for i, view in enumerate(viewpoint_stack):
        if i == 0:
            H = view.image_height * r
            W = view.image_width * r

        f_x = fov2focal(view.FoVx, W)
        f_y = fov2focal(view.FoVy, H)
        K = np.array([
                [f_x, 0, W / 2],
                [0, f_y, H / 2],
                [0, 0, 1]
            ])
        Ks.append(K)

        c2ws_adjust.append(np.linalg.inv(view.world_view_transform.T.cpu().numpy()))
        try:
            c2ws_org.append(np.linalg.inv(view.world_view_transform_origin.T.cpu().numpy()))
        except:
            pass
        file_names.append(view.image_name)
        
    write_transformsfile(H, W, Ks, c2ws=c2ws_adjust, file_names=file_names, save_path=save_path, stuff=stuff) 


def write_transformsfile(H, W, Ks, c2ws, file_names, save_path, stuff = None):
    frames = []
    # 读取图片名以及对应的pose
    for i, line in enumerate(c2ws):
        if i == 0:
            data = dict(
                fl_x=Ks[i][0][0],
                fl_y=Ks[i][1][1],
                k1=0,
                k2=0,
                k3=0,
                k4=0,
                p1=0,
                p2=0,
                is_fisheye=False,
                cx=Ks[i][0][2],
                cy=Ks[i][1][2],
                w=W,
                h=H,
                aabb_scale=16,
            )
        frame = {}
        frame["file_path"] = f"images/{file_names[i]}"
        # Transform matrix to nerfstudio format. Please refer to the documentation for coordinate system conventions.
        frame["transform_matrix"] = [j.tolist() for j in c2ws[i]]
        frames.append(frame)
    data["frames"] = frames
    if stuff is not None:
        with open(f"{save_path}/transforms_{stuff}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    else:
        with open(f"{save_path}/transforms.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)   


DELTA_T_RANGE = 1
DELTA_ANGLE_RANGE = 0.5
def add_delta_to_RT(R, T):
    """
    为R,T矩阵添加高斯噪声
    """
    np.random.seed(2024)
    delta_t = np.random.uniform(-DELTA_T_RANGE, DELTA_T_RANGE, size=(3,))
    delta_angle = np.random.uniform(-DELTA_ANGLE_RANGE, DELTA_ANGLE_RANGE, size=(3,))
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(delta_angle[0]), -np.sin(delta_angle[0])],
                   [0, np.sin(delta_angle[0]), np.cos(delta_angle[0])]])
    RY = np.array([[np.cos(delta_angle[1]), 0, np.sin(delta_angle[1])],
                     [0, 1, 0],
                     [-np.sin(delta_angle[1]), 0, np.cos(delta_angle[1])]])
    Rz = np.array([[np.cos(delta_angle[2]), -np.sin(delta_angle[2]), 0],
                   [np.sin(delta_angle[2]), np.cos(delta_angle[2]), 0],
                   [0, 0, 1]])
    delta_rotation = torch.from_numpy(Rz @ RY @ Rx).to(R.device)
    delta_t = torch.from_numpy(delta_t).to(T.device)
    R @= delta_rotation
    T += delta_t
    return R, T
