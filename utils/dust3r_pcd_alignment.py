import sys
sys.path.append('./mast3r/dust3r/')
import os

import cv2
from dust3r.cloud_opt.base_opt import global_alignment_loop

from sfm_free_utils.optimization_sRT import *
from sfm_free_utils.utils import *
import torch.nn.functional as F

import numpy as np
import open3d as o3d
from glob import glob
from pathlib import Path
import torch
import math
import argparse
import logging
   

class Pcd_Global_Alignment(nn.Module):
    """ Optimize a global scene, given a list of keyframes pcd.
    Graph node: keyframes pcd
    Graph edges: sRT matrix 
    """
    def __init__(self, scene, device='cuda', camera_align=True):
        super(Pcd_Global_Alignment, self).__init__()

        self.device = device
        self.verbose = True
        self.camera_align = camera_align
        # 数据读取与构造
        self.meta_list = []
        print("read mata data...")
        for idx, block in enumerate(scene['keyframe']):
            print(f"read img pair {block}")
            self.meta_list.append(self.read_meta(scene, idx))

        self.correlation_matrix = self.get_correlation(self.meta_list)
        # 这里包含了所有我们需要的数据信息，现在只需要设计mutitask任务来优化可学习参数
        self.s_t_pairs, self.loopback_idx, self.loopback_pairs, init_poses, self.s_t_camera_pairs, self.loopback_camera_pairs = self.get_pcd_pairs(self.correlation_matrix, self.meta_list)
        self.init_model(init_poses)

    def init_model(self, init_poses=None):
        # 标志位
        self.delta = True
        # add learnable tensor 
        print("init learnable tensor")
        self.R = []
        self.T = []
        self.scale = nn.ParameterList()
        self.rot_delta = nn.ParameterList()
        self.trans_delta = nn.ParameterList()
        assert init_poses is not None, "Good initialization is a must for this task!" 
        for i in init_poses:
            # self.scale.append(nn.Parameter(
            #                 torch.ones(3, requires_grad=True, device=self.device)
            #                 ))
            self.scale.append(nn.Parameter(
                            torch.ones(1, requires_grad=True, device=self.device)
                            ))
            self.R.append(torch.tensor(i[:3, :3]).to(self.device))
            self.T.append(torch.tensor(i[:3, 3]).to(self.device))
            # 建模扰动因子
            self.rot_delta.append(nn.Parameter(
                torch.zeros(3, requires_grad=True, device=self.device)
            ))
            self.trans_delta.append(nn.Parameter(
                torch.zeros(3, requires_grad=True, device=self.device)
                ))

    @staticmethod
    def read_meta(scene, idx):

        block_pcd = {Path(k).name :v for k, v in zip(scene['keyframe'][idx], scene['pcds'][idx])}
        block_conf = {Path(k).name :v for k, v in zip(scene['keyframe'][idx], scene['confs'][idx])} 
        block_pose = {Path(k).name :v for k, v in zip(scene['keyframe'][idx], scene['pose'][idx])}
        meta = dict(
            pcd=block_pcd,
            conf=block_conf,
            pose=block_pose)
        
        return meta
    
    @staticmethod
    def get_correlation(meta_list):

        correlation_matrix = [[0 for _ in range(len(meta_list))] for _ in range(len(meta_list))]


        for i in range(len(meta_list)-1):
            for j in range(i + 1, len(meta_list)):
                if len(list(set(meta_list[i]['pcd'].keys()).intersection(set(meta_list[j]['pcd'].keys())))) == 0: continue 
                correlation_matrix[i][j] = list(set(meta_list[i]['pcd'].keys()).intersection(set(meta_list[j]['pcd'].keys())))

        return correlation_matrix

    @staticmethod
    def sample_pcd(s_pcd, t_pcd, num_points=10000):
        # Generate random indices
        num_total_points = s_pcd.shape[0]

        if num_total_points > num_points:

            random_indices = np.random.choice(num_total_points, size=num_points, replace=False)
            # Select sampled points using random indices
            sampled_s_pcd = s_pcd[random_indices, :]
            sampled_t_pcd = t_pcd[random_indices, :]
            return sampled_s_pcd, sampled_t_pcd
        
        else:
            return s_pcd, t_pcd
   
    def get_pcd_pairs(self, correlation_matrix, meta_list, debug=False, sample=False):
        """"
        获取重叠区域的pcd，构建pairs,保证构造的paris的顺序，
        """
        target_idx, source_idx = np.where(np.array(correlation_matrix, dtype=object) != 0)

        s_t_camera_pairs = []
        # 非回环时的pairs
        s_t_pairs = []
        # 记录回环出现的起始节点，从而通过节点索引构建约束
        loopback_idx = []
        # 回环paris中放置了通过更多的转换阵组合才能描述的映射关系
        loopback_pairs = []
        loopback_camera_pairs = []
        # 根据相同关键帧计算s到t的初始化变换阵
        init_poses = []

        for t, s in zip(target_idx, source_idx):
            print(f"make pairs for target scene {t} - source scene {s}")
            keyframes = correlation_matrix[t][s]
            if isinstance(keyframes, list):
                target_pcd_merge = []
                source_pcd_merge = []
                for i in keyframes:
                    # 查询关键帧在相机坐标下的点云
                    # # 这里需要注意的是两张图像推理得到的conf是不同的，因此需要将两者的conf_masks进行叠加从而得到新的masks
                    merge_masks = meta_list[t]['conf'][i] & meta_list[s]['conf'][i] 
                    target_pcd_merge.append(meta_list[t]['pcd'][i][merge_masks])
                    source_pcd_merge.append(meta_list[s]['pcd'][i][merge_masks])

                target_pcd_merge = np.concatenate(target_pcd_merge)
                source_pcd_merge = np.concatenate(source_pcd_merge)
                if target_pcd_merge.shape[0] == 0:
                    print(merge_masks.sum())
                # sample pcd
                if sample:
                    target_pcd_merge, source_pcd_merge = self.sample_pcd(target_pcd_merge, source_pcd_merge)

            else:
                # 查询关键帧在相机坐标下的点云
                # # 这里需要注意的是两张图像推理得到的conf是不同的，因此需要将两者的conf_masks进行叠加从而得到新的masks
                merge_masks = meta_list[t]['conf'][keyframes] & meta_list[s]['conf'][keyframes] 
                # load
                target_pcd_merge = meta_list[t]['pcd'][keyframes][merge_masks]
                source_pcd_merge = meta_list[s]['pcd'][keyframes][merge_masks]

            print(f"keyframes pcd shape with scene_{t}_{s}: {target_pcd_merge.shape}")

            # 转换为齐次表示
            suffix = np.ones(target_pcd_merge.shape[0])[:, None]
            target_pcd_merge = np.hstack([target_pcd_merge, suffix])
            source_pcd_merge = np.hstack([source_pcd_merge, suffix])

            if debug:
                pcd_s = o3d.geometry.PointCloud()
                pcd_s.points = o3d.utility.Vector3dVector(meta_list[s]['pcd'][i][merge_masks])
                o3d.io.write_point_cloud(f"source_{s}.ply", pcd_s)

                pcd_t = o3d.geometry.PointCloud()
                pcd_t.points = o3d.utility.Vector3dVector(meta_list[t]['pcd'][i][merge_masks])
                o3d.io.write_point_cloud(f"target_{t}.ply", pcd_t)

            if s - t > 1:
                loopback_idx.append((s, t))
                loopback_pairs.append([torch.from_numpy(source_pcd_merge).type(torch.float32).to(self.device), torch.from_numpy(target_pcd_merge).type(torch.float32).to(self.device)])

                # add camera point 
                t_keyframes_poses = np.concatenate([meta_list[t]['pose'][i][None, :, 3] for i in keyframes])
                s_keyframes_poses = np.concatenate([meta_list[s]['pose'][i][None, :, 3] for i in keyframes])
                loopback_camera_pairs.append([torch.from_numpy(s_keyframes_poses).type(torch.float32).to(self.device), torch.from_numpy(t_keyframes_poses).type(torch.float32).to(self.device)])

            else:
                s_t_pairs.append([torch.from_numpy(source_pcd_merge).type(torch.float32).to(self.device), torch.from_numpy(target_pcd_merge).type(torch.float32).to(self.device)])

                # add camera point 
                t_keyframes_poses = np.concatenate([meta_list[t]['pose'][i][None, :, 3] for i in keyframes])
                s_keyframes_poses = np.concatenate([meta_list[s]['pose'][i][None, :, 3] for i in keyframes])
                s_t_camera_pairs.append([torch.from_numpy(s_keyframes_poses).type(torch.float32).to(self.device), torch.from_numpy(t_keyframes_poses).type(torch.float32).to(self.device)])

                # init pose compute , 不要构造回环分支的
                t_keyframes_pose = meta_list[t]['pose'][keyframes[0]]
                s_keyframes_pose = meta_list[s]['pose'][keyframes[0]]
                init_pose =  np.dot(t_keyframes_pose, np.linalg.inv(s_keyframes_pose))
                # 描述的是 t = T @ s
                init_poses.append(init_pose)

        return s_t_pairs, loopback_idx, loopback_pairs, init_poses, s_t_camera_pairs, loopback_camera_pairs
    
    # 定义
    def get_transform_st(self, st_idx):

        """
        trans_martix = [ sR, t
                        0, 1]

        """

        trans = torch.eye(4, device=self.device)
        tau = torch.cat([self.trans_delta[st_idx], self.rot_delta[st_idx]], axis=0)
        delta_r, delta_t = SE3_exp(tau)
        
        # trans[:3, :3] = self.scale[st_idx] * (delta_r @ self.R[st_idx])
        if self.scale[st_idx].shape[0] == 1:
            trans[:3, :3] = torch.diag(self.scale[st_idx].repeat(3)) @ (delta_r @ self.R[st_idx])
        else:
            trans[:3, :3] = torch.diag(self.scale[st_idx]) @ (delta_r @ self.R[st_idx])
        trans[:3, 3] = self.T[st_idx] + delta_t

        return trans

    def get_transform_loopback(self, loopback_idx):
        """
        trans_martix = [ sR, t
                        0, 1]


        self.loopback_idx

        假设s_idx = 0, e_idx=2
        那么回环的trans = T0 @ T1 @ T2
        """

        trans = torch.eye(4, device=self.device)

        for i in range(loopback_idx[0]-1, loopback_idx[1]-1, -1):
            trans_tmp = torch.eye(4, device=self.device)
            tau = torch.cat([self.trans_delta[i], self.rot_delta[i]], axis=0)
            delta_r, delta_t = SE3_exp(tau)
            
            # trans_tmp[:3, :3] = self.scale[i] * (delta_r @ self.R[i])
            if self.scale[i].shape[0] == 1:
                trans_tmp[:3, :3] = torch.diag(self.scale[i].repeat(3)) @ (delta_r @ self.R[i])
            else:
                trans_tmp[:3, :3] = torch.diag(self.scale[i]) @ (delta_r @ self.R[i])
            trans_tmp[:3, 3] = self.T[i] + delta_t

            trans = trans_tmp @ trans 

        return trans

    def update_RT(self, R, t, i):
        self.R[i] = R.to(device=self.device)
        self.T[i] = t.to(device=self.device)


    def update_trans(self, converged_threshold=1e-4):
        for i in range(len(self.trans_delta)):
            tau = torch.cat([self.trans_delta[i].data, self.rot_delta[i].data], axis=0)

            delta_r, delta_t = SE3_exp(tau)
            new_R = delta_r @ self.R[i]
            new_T = self.T[i] + delta_t

            converged = tau.norm() < converged_threshold
            self.update_RT(new_R, new_T, i)

            self.rot_delta[i].data.fill_(0)
            self.trans_delta[i].data.fill_(0)

        return converged

    def forward(self):
        loss = 0
        # st
        for idx, pairs in enumerate(self.s_t_pairs):
            trans = self.get_transform_st(idx)
            y = (trans @ pairs[0].T).T
            # y_st.append(y)
            # gt_st.append(pairs[1])
            # if self.Euclidean_distance_loss(y, pairs[1]) > 10:
            #     print(self.Euclidean_distance_loss(y, pairs[1]))
            #     continue
            # else:
            loss += self.Euclidean_distance_loss(y, pairs[1]).clamp(0, 5)
        
        #loopback
        for idx, pairs in zip(self.loopback_idx, self.loopback_pairs):
            trans = self.get_transform_loopback(idx)
            y = (trans @ pairs[0].T).T
            # y_loopback.append(y)
            # gt_loopback.append(pairs[1])
            loss += 0.1 * self.Euclidean_distance_loss(y, pairs[1])

        if self.camera_align:
            for idx, pairs in enumerate(self.s_t_camera_pairs):
                trans = self.get_transform_st(idx)
                y = (trans @ pairs[0].T).T
                # y_st.append(y)
                # gt_st.append(pairs[1])
                loss +=  self.Euclidean_distance_loss(y, pairs[1])
        
            # loopback
            for idx, pairs in zip(self.loopback_idx, self.loopback_camera_pairs):
                trans = self.get_transform_loopback(idx)
                y = (trans @ pairs[0].T).T
                # y_loopback.append(y)
                # gt_loopback.append(pairs[1])
                loss += 0.01 * self.Euclidean_distance_loss(y, pairs[1])

        return loss

    @staticmethod
    def Euclidean_distance_loss(in1, in2):
        loss = torch.mean(torch.norm(in1 - in2, dim=1)) 
        return loss

    @torch.cuda.amp.autocast(enabled=False)
    def compute_global_alignment(self, **kw):

        return global_alignment_loop(self, **kw)


    def get_result(self):
        print("scale = ", [i.data.cpu().numpy().tolist() for i in self.scale] if isinstance(self.scale, nn.ParameterList) else self.scale.data.cpu().numpy().tolist())            
        print("R = ", [i.data.cpu().numpy().tolist() for i in self.R] if isinstance(self.R, list) else self.R.data.cpu().numpy())
        print("T = ", [i.data.cpu().numpy().tolist() for i in self.T] if isinstance(self.R, list) else self.T.data.cpu().numpy())

        return [i.data.cpu().numpy().tolist() for i in self.scale] if isinstance(self.scale, nn.ParameterList) else self.scale.data.cpu().numpy().tolist(), \
                [i.data.cpu().numpy().tolist() for i in self.R] if isinstance(self.R, list) else self.R.data.cpu().numpy(),\
                [i.data.cpu().numpy().tolist() for i in self.T] if isinstance(self.R, list) else self.T.data.cpu().numpy()
