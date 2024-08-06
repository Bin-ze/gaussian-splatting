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

import os
import sys
sys.path.append('./mast3r/dust3r/')
sys.path.append('./mast3r/')
from mast3r.fast_nn import fast_reciprocal_NNs
from mast3r.model import AsymmetricMASt3R

from PIL import Image
from typing import NamedTuple
from utils.graphics_utils import getWorld2View1, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB

from utils.dust3r_pcd_alignment import Pcd_Global_Alignment
from scene.gaussian_model import BasicPointCloud

from utils.graphics_utils import fov2focal
from dust3r.cloud_opt import fast_pnp, estimate_focal
from dust3r.utils.image import load_images, rgb
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs

from sfm_free_utils.optimization_sRT import *
from sfm_free_utils.utils import *
import torch.nn.functional as F

import numpy as np
import open3d as o3d
from glob import glob
import torch
import math
import cv2

MODEL_PATH = "/home/guozebin/work_code/sfm-free-gaussian-splatting/ckpt/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
Mast3R_MODEL_PATH = "/home/guozebin/work_code/sfm-free-gaussian-splatting/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"


class SceneInfo(NamedTuple):
    point_cloud: list
    train_cameras: list
    alignment_trans: list
    nerf_normalization: dict

class CameraInfo(NamedTuple):
    uid: int
    R: list
    T: list
    FovY: np.array
    FovX: np.array
    image: np.array
    mask: np.array
    depth: np.array
    image_path: str
    image_name: str
    width: int
    height: int

def storePly_BasicPointCloud(path, pcd: BasicPointCloud):
    '''
    points : np.array
    colors : np.array
    normals : np.array

    '''
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    xyz = np.vstack(pcd.points)
    rgb = np.vstack(pcd.colors) * 255
    normals = np.vstack(pcd.normals)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal
    
    res = []
    for cam in cam_info:
        cam_centers = []
        for i in range(len(cam.R)):
            W2C = getWorld2View1(cam.R[i], cam.T[i])
            C2W = np.linalg.inv(W2C)
            cam_centers.append(C2W[:3, 3:4])

        center, diagonal = get_center_and_diag(cam_centers)
        radius = diagonal * 1.1

        translate = -center
        res.append({"translate": translate, "radius": radius})
    return res


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def make_st_pcd_pairs(pcds, sparse_mask):
    '''
    使用重叠关键帧构建优化相似矩阵所需的source和target
    source: 后一个局部场景第一帧的点云
    target: 前一个局部场景第二帧的点云
    '''
    s_t_pairs = []

    for i in range(len(pcds) - 1):
        filter_mask = sparse_mask[i][-1] & sparse_mask[i+1][0]
        source_pcd = pcds[i+1][0][filter_mask]
        target_pcd = pcds[i][-1][filter_mask]
        # 转换为齐次表示
        suffix = np.ones(target_pcd.shape[0])[:, None]
        target_pcd_merge = np.hstack([target_pcd, suffix])
        source_pcd_merge = np.hstack([source_pcd, suffix])

        s_t_pairs.append([torch.from_numpy(source_pcd_merge).type(torch.float32).to('cuda'), torch.from_numpy(target_pcd_merge).type(torch.float32).to('cuda')])

    return s_t_pairs

def fetchPly(pcds, colors, mask=None, sample_point=50000):
    '''
    以pairs为单位，每对pairs代表一个全局一致的local scene
    '''
    if mask is not None:
        positions = [i[mask[idx]] for idx, i in enumerate(pcds)]
        colors = [i[mask[idx]] for idx, i in enumerate(colors)]
    else:
        positions = [i.reshape(-1, 3) for idx, i in enumerate(pcds)]
        colors = [i.reshape(-1, 3) for idx, i in enumerate(colors)]

    num_pts = [i.shape[0] for i in positions]
    normals = [np.zeros((i, 3)) for i in num_pts]

    sample_positions = []
    sample_colors = []
    sample_normals = []   
    for idx, meta in enumerate(zip(positions, colors, normals)):
        position, color, normal = meta
        if idx == 0:
            print(f"在第一个paris保留 {position.shape[0]//2} points")
            sub_ind = np.random.choice(position.shape[0], position.shape[0] // 2, replace=False)
            position = position[sub_ind]  # numpy array
            color = color[sub_ind]  # numpy array
            normal = normal[sub_ind]

            sample_positions.append(position)
            sample_colors.append(color)
            sample_normals.append(normal)
            continue
        if position.shape[0] > sample_point:
            print(f"初始化点云密集！进行随机采样到 {sample_point} points")
            sub_ind = np.random.choice(position.shape[0], sample_point, replace=False)
            position = position[sub_ind]  # numpy array
            color = color[sub_ind]  # numpy array
            normal = normal[sub_ind]
        sample_positions.append(position)
        sample_colors.append(color)
        sample_normals.append(normal)
        
    
    return [BasicPointCloud(points=i, colors=j, normals=k) for i, j, k in zip(sample_positions, sample_colors, sample_normals)]

def merge_BasicPointCloud(pcd1, pcd2):

    points1 = pcd1.points
    points2 = pcd2.points

    colors1 = pcd1.colors
    colors2 = pcd2.colors

    normals1 = pcd1.normals
    normals2 = pcd2.normals


    return BasicPointCloud(points=np.concatenate([points1, points2]), \
            colors=np.concatenate([colors1, colors2]), normals=np.concatenate([normals1, normals2]))


def parser_res(output):
    view1_pts3d = output['pred1']['pts3d']
    view1_conf = output['pred1']['conf']
    view1_img = output['view1']['img']
    view2_pts3d = output['pred2']['pts3d_in_other_view']
    view2_conf = output['pred2']['conf']
    view2_img = output['view2']['img']

    return view1_pts3d, view1_conf, view1_img, view2_pts3d, view2_conf, view2_img


def parser_res_mast3r(output):

    view1_pts3d = output['pred1']['pts3d']
    view1_conf = output['pred1']['conf']
    view1_img = output['view1']['img']

    view2_pts3d = output['pred2']['pts3d_in_other_view']
    view2_conf = output['pred2']['conf']
    view2_img = output['view2']['img']

    matches_im0, matches_im1 = find_valid_matches(output)

    return view1_pts3d, view1_conf, view1_img, view2_pts3d, view2_conf, view2_img, matches_im0, matches_im1

def find_valid_matches(output, device="cuda"):

    desc1 = output['pred1']['desc'].squeeze(0).detach()
    desc2 = output['pred2']['desc'].squeeze(0).detach()
    # find 2D-2D matches between the two images
    matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                   device=device, dist='dot', block_size=2**13)

    # ignore small border around the edge
    H0, W0 = output['view1']['true_shape'][0]
    valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
        matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

    H1, W1 = output['view2']['true_shape'][0]
    valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
        matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

    valid_matches = valid_matches_im0 & valid_matches_im1
    matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

    return matches_im0, matches_im1


def export_pcd(scene, idx, debug=False):
    pts3d, confidence_masks, imgs = scene['pcds'][idx], scene['confs'][idx], scene['colors'][idx]
    pts = np.concatenate([p[m] for p, m in zip(pts3d, confidence_masks)])
    col = np.concatenate([p[m] for p, m in zip(imgs, confidence_masks)])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(col)
    if debug:
        o3d.io.write_point_cloud(f"points3d.ply", pcd)
    
    return pcd

def pose_trans(pose, trans, R):
    pose[:, :3, :3] = np.array(R) @ pose[:, :3, :3]
    pose[:, :, 3] = (trans @ pose[:, :, 3, None])[:, ..., 0]
    
    return pose

def project_pcd_to_depth(pcds, focals, poses):
    '''
    project:
    camera_pcd = w2c @ pcds
    img_pcd = K @ camera_pcd
    '''
    def project_pcd_to_depth_single(pcd, focal, pose):
        h, w = pcd.shape[:2]
        world_pcd = pcd.reshape(-1, 3)
        suffix = np.ones(world_pcd.shape[0])[:, None]
        world_pcd_homogeneous = np.hstack([world_pcd, suffix])

        camrea_pcd_homogeneous = (np.linalg.inv(pose) @ world_pcd_homogeneous.T).T

        K = np.array([
            [focal, 0, w/2],
            [0, focal, h/2],
            [0, 0, 1]
        ])

        image_pcd = (K @ camrea_pcd_homogeneous[:, :3].T).T
        
        z = image_pcd[:, 2:3].reshape((h, w))

        return z
    
    depths = []
    for pcd, focal, pose in zip(pcds, focals, poses): # seq
        depth = []
        for i, j in zip(pcd, pose): # pairs
            depth.append(project_pcd_to_depth_single(i, focal, j)[None])
        depths.append(np.concatenate(depth))

    return depths

def local_2_global_trans(scale, R, T):
    trans_s2t = []
    trans_s2t_R = []
    space_len = len(scale) + 1
    for i in range(space_len):
        trans = np.eye(4)
        r = np.eye(3)
        for j in range(i):
            t = get_transform(scale[j], R[j], T[j])
            trans = trans @ t
            r = r @ np.array(R[j])

        trans_s2t.append(trans.astype(np.float32))
        trans_s2t_R.append(r.astype(np.float32))

    return trans_s2t, trans_s2t_R

def decompose_similarity_matrix_nonuniform(M):
    # 提取旋转矩阵
    A = M[:3, :3]

    # 计算缩放向量s
    s = np.linalg.norm(A, axis=1)

    # # 对每一行除以对应的缩放系数
    # R = A / s[:, np.newaxis]

    # # 提取平移向量t
    # t = M[:3, 3]

    return s[0]

def readSFM_FreeCameras(c2ws, focals, h, w, pairs, masks=None, depths=None):
    
    cam_infos = []
    for idx, key in enumerate(c2ws):
        sys.stdout.write('\r')
        sys.stdout.write("Reading camera pairs {}/{}".format(idx+1, len(c2ws)))
        sys.stdout.flush()
        
        R = [i[:3, :3] for i in key]
        T = [np.linalg.inv(i)[:3, 3] for i in key]
        
        image = [Image.open(image_path) for image_path in pairs[idx]]
        if masks is not None:
            mask = [Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)  for mask in masks[idx]]
        if depths is not None:
            depth = [Image.fromarray(depth).resize((w, h), Image.NEAREST)  for depth in depths[idx]]
        FovX = focal2fov(focals[idx], w)
        FovY = focal2fov(fov2focal(FovX, w), h)
        
        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                    image_path=pairs[idx], image_name=[Path(i).name for i in pairs[idx]], width=w, height=h, mask=mask, depth=depth))

    return cam_infos      
        
    
def readSFMFreeSceneInfo(dataset_path, min_conf_thr=3, img_fov=90, focal_known=False, scale=20, device='cuda', fps=1, fast=False):
    # instance model
    model = AsymmetricCroCo3DStereo.from_pretrained(MODEL_PATH).to(device)
    # 对于整个数据集，首先排序，然后拆分
    img_fps = sorted(glob(f'{dataset_path}/images/*'))
    img_fps = img_fps[::fps]
    # 计算原始的图像wh
    org_h, org_w = cv2.imread(img_fps[0]).shape[:2]    

    split_img = []
    for i in range(len(img_fps) - 1):
        
        split_img.append([img_fps[i], img_fps[i+1]])
    
    # record inference result 
    infer_result = []
    
    for img_pair in split_img:
        images = load_images(img_pair, size=512, square_ok=True)
        # 计算读取之后的图像wh
        h, w = images[0]['true_shape'][0]
        pairs = make_pairs(images, scene_graph='oneref', symmetrize=False)
        output = inference(pairs=pairs, model=model, device=device, batch_size=1)
        infer_result.append(output)
        
    del model

    c2ws = []
    focals = []
    pts3ds = []
    dense_mask = []
    colors = []

    for idx, o in enumerate(infer_result):
        view1_pts3d, view1_conf, view1_img, view2_pts3d, view2_conf, view2_img = parser_res(o)
        # format
        pts3d = torch.cat([view1_pts3d, view2_pts3d])
        conf = torch.cat([view1_conf, view2_conf])
        imgs = np.concatenate([rgb(view1_img), rgb(view2_img)])
        pose = []
        focal = []

        for v, m in zip(pts3d, conf):
            # msk = m > min_conf_thr
            top_1000_thr = torch.topk(m.reshape(-1), 1000).values.min()
            msk = m > top_1000_thr
            if img_fov:
                im_focals = fov2focal(math.radians(img_fov), 512)
                print(f"using knowned focals {im_focals}")
            elif focal_known:
                im_focals = float(focal_known) * (w / org_w)
                print(f"using knowned focals {im_focals}")
            else:
                im_focals = estimate_focal(v)   
                print(f"using estimate focals {im_focals}")
            try:
                f, P = fast_pnp(v, im_focals, msk=msk, device=device, niter_PnP=10)
                # print(f, P)
                pose.append(P[None].cpu())
                focal.append(torch.tensor([[f]]))
            except:
                print(f'init pose estimate fail {split_img[idx]},  using pose of the previous frame')
                try:
                    pose.append(pose[-1])
                except:
                    pose.append(torch.eye(4)[None])

                focal.append(torch.tensor([[im_focals]]))
            
        pose = torch.cat(pose)
        focal = torch.cat(focal)

        # rescale 
        pose[:, :3, 3] *= scale
        pts3d *= scale
        focal = focal.mean() * (org_w / w)
        
        top_thr = torch.topk(conf.reshape(-1), int(conf.reshape(-1).shape[0] * 0.3)).values.min()
        c2ws.append(pose.numpy())
        focals.append(focal.numpy())
        pts3ds.append(pts3d.numpy())
        dense_mask.append((conf > top_thr).numpy())
        # sparse_mask.append((conf > top_1000_thr).numpy())
        colors.append(imgs)
    # 深度监督对该任务可以说是至关重要的：在视角较少时，十分容易出现过拟合
    # 这里准备深度图用于后续的监督： 深度图不需要尺度，只保留相对深度即可
    depth = project_pcd_to_depth(pts3ds, [i * (w/org_w) for i in focals], c2ws)

    # Calculate similarity transformations between different local scenes
    scene = dict(
        pose=c2ws,
        pcds=pts3ds,
        confs=dense_mask,
        keyframe=split_img,
        )
    align_net = Pcd_Global_Alignment(scene=scene, camera_align=False)
    if not fast:
        align_net.compute_global_alignment(lr=0.01, niter=300, schedule='cosine')    
        s, R, T = align_net.get_result()
    else:
        s, R, T = align_net.get_result()
    
    del align_net
    # 将其转换为局部到全局的格式
    trans_s2t, trans_s2t_R = local_2_global_trans(s, R, T)
    # print(trans_s2t)
    alignment_trans = dict(
        trans_s2t=trans_s2t,
        trans_s2t_R=trans_s2t_R
    )
    # alignment depth scale to global space
    for M, i in zip(trans_s2t, depth):
        i *= decompose_similarity_matrix_nonuniform(M)
        
    pcds = fetchPly(pts3ds, colors)
    cam_infos = readSFM_FreeCameras(c2ws, focals, org_h, org_w, split_img, dense_mask, depth)


    nerf_normalization = getNerfppNorm(cam_infos)
    # print(nerf_normalization)
    scene_info = SceneInfo(point_cloud=pcds,
                           train_cameras=cam_infos,
                           alignment_trans=alignment_trans,
                           nerf_normalization=nerf_normalization)
    return scene_info
            
def readSFMFreeSceneInfo_v1(dataset_path, min_conf_thr=3, img_fov=90, focal_known=False, device='cuda', fps=3, fast=False):
    # instance model
    model = AsymmetricMASt3R.from_pretrained(Mast3R_MODEL_PATH).to(device)
    # 对于整个数据集，首先排序，然后拆分
    img_fps = sorted(glob(f'{dataset_path}/images/*'))
    img_fps = img_fps[::fps]
    # 计算原始的图像wh
    org_h, org_w = cv2.imread(img_fps[0]).shape[:2]    

    split_img = []
    for i in range(len(img_fps) - 1):
        
        split_img.append([img_fps[i], img_fps[i+1]])
    
    # record inference result 
    infer_result = []
    
    for img_pair in split_img:
        images, _ = load_images(img_pair, size=512, square_ok=True)
        # 计算读取之后的图像wh
        h, w = images[0]['true_shape'][0]
        pairs = make_pairs(images, scene_graph='oneref', symmetrize=False)
        output = inference(pairs=pairs, model=model, device=device, batch_size=1)
        infer_result.append(output)
        
    del model

    c2ws = []
    focals = []
    pts3ds = []
    dense_mask = []
    colors = []

    for idx, o in enumerate(infer_result):
        view1_pts3d, view1_conf, view1_img, view2_pts3d, view2_conf, view2_img, matches_im0, matches_im1 = parser_res_mast3r(o)
        # format
        pts3d = torch.cat([view1_pts3d, view2_pts3d])
        conf = torch.cat([view1_conf, view2_conf])
        imgs = np.concatenate([rgb(view1_img), rgb(view2_img)])
        pose = []
        focal = []

        for v, m, match in zip(pts3d, conf, (matches_im0, matches_im1)):
            # msk = m > min_conf_thr
            msk = torch.zeros_like(m).type(torch.bool)
            # using match inform
            x_coords = match[:, 0]
            y_coords = match[:, 1]
            msk[y_coords, x_coords] = True

            if img_fov:
                im_focals = fov2focal(math.radians(img_fov), 512)
                print(f"using knowned focals {im_focals}")
            elif focal_known:
                im_focals = float(focal_known) * (w / org_w)
                print(f"using knowned focals {im_focals}")
            else:
                im_focals = estimate_focal(v)   
                print(f"using estimate focals {im_focals}")
            try:
                f, P = fast_pnp(v, im_focals, msk=msk, device=device, niter_PnP=10)
                # print(f, P)
                pose.append(P[None].cpu())
                focal.append(torch.tensor([[f]]))
            except:
                print(f'init pose estimate fail {split_img[idx]},  using pose of the previous frame')
                try:
                    pose.append(pose[-1])
                except:
                    pose.append(torch.eye(4)[None])

                focal.append(torch.tensor([[im_focals]]))
            
        pose = torch.cat(pose)
        focal = torch.cat(focal)
        focal = focal.mean() * (org_w / w)
        
        top_thr = torch.topk(conf.reshape(-1), int(conf.reshape(-1).shape[0] * 0.8)).values.min()
        c2ws.append(pose.numpy())
        focals.append(focal.numpy())
        pts3ds.append(pts3d.numpy())
        dense_mask.append((conf > top_thr).numpy())
        # sparse_mask.append((conf > top_1000_thr).numpy())
        colors.append(imgs)
    # 深度监督对该任务可以说是至关重要的：在视角较少时，十分容易出现过拟合
    # 这里准备深度图用于后续的监督： 深度图不需要尺度，只保留相对深度即可
    depth = project_pcd_to_depth(pts3ds, [i * (w/org_w) for i in focals], c2ws)

    # Calculate similarity transformations between different local scenes
    scene = dict(
        pose=c2ws,
        pcds=pts3ds,
        confs=dense_mask,
        keyframe=split_img,
        )
    align_net = Pcd_Global_Alignment(scene=scene, camera_align=False)
    if not fast:
        align_net.compute_global_alignment(lr=0.01, niter=300, schedule='cosine')    
        s, R, T = align_net.get_result()
    else:
        s, R, T = align_net.get_result()
    
    del align_net
    # 将其转换为局部到全局的格式
    trans_s2t, trans_s2t_R = local_2_global_trans(s, R, T)
    # print(trans_s2t)
    alignment_trans = dict(
        trans_s2t=trans_s2t,
        trans_s2t_R=trans_s2t_R
    )
    # alignment depth scale to global space
    for M, i in zip(trans_s2t, depth):
        i *= decompose_similarity_matrix_nonuniform(M)
        
    pcds = fetchPly(pts3ds, colors)
    cam_infos = readSFM_FreeCameras(c2ws, focals, org_h, org_w, split_img, dense_mask, depth)


    nerf_normalization = getNerfppNorm(cam_infos)
    # print(nerf_normalization)
    scene_info = SceneInfo(point_cloud=pcds,
                           train_cameras=cam_infos,
                           alignment_trans=alignment_trans,
                           nerf_normalization=nerf_normalization)
    return scene_info
    
sceneLoadTypeCallback = {
    "SFMFree_dust3r": readSFMFreeSceneInfo,
    "SFMFree_mast3r": readSFMFreeSceneInfo_v1,
}