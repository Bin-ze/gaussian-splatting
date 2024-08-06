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
import random
import json
import torch
import open3d as o3d
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.dataset_readers_sfm_free import sceneLoadTypeCallback, storePly_BasicPointCloud
from scene.gaussian_model import GaussianModel
from scene.gaussian_model_sfm_free import GaussianModel_SFMFree, BasicPointCloud
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, cameraList_from_sfmfree_camInfos
from sfm_free_utils.utils import pcd_trans, merge_pcd

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms.json")) or os.path.exists(os.path.join(self.model_path, "transforms.json")):
            print("Found transforms.json file, assuming NeRFstudio data set!")
            scene_info = sceneLoadTypeCallbacks["NeRFstudio"](args.source_path, args.eval, model_path=self.model_path)        
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

class Scene_Free:

    gaussians : GaussianModel_SFMFree

    def __init__(self, args : ModelParams, gaussians : GaussianModel_SFMFree, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.source_path = args.source_path
        self.loaded_iter = None
        self.gaussians = gaussians

        self.train_cameras = {}

        # dataset_path, min_conf_thr=3, img_fov=90, scale=20
        if args.mast3r:
            self.scene_info = sceneLoadTypeCallback["SFMFree_mast3r"](args.source_path, args.min_conf_thr, args.img_fov, args.focal_known)
        else:
            self.scene_info = sceneLoadTypeCallback["SFMFree_dust3r"](args.source_path, args.min_conf_thr, args.img_fov, args.focal_known, args.scene_scale)
        self.cameras_extent = [i["radius"] for i in self.scene_info.nerf_normalization]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_sfmfree_camInfos(self.scene_info.train_cameras, resolution_scale, args)

        # using first local scene init gaussian
        self.gaussians.create_from_pcd(self.scene_info.point_cloud[0], self.cameras_extent[0])
        # 
        storePly_BasicPointCloud(os.path.join(args.source_path, 'points3d.ply'), self.scene_info.point_cloud[0])

        self.prior_pcd = self.trans_BasicPointCloud(self.scene_info.point_cloud[0])

    def save(self, iteration=None):
        if iteration is not None:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
            self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        else:
            self.gaussians.save_ply(os.path.join(self.model_path, "point_cloud_Progressive.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def alignment_global_scene_pose(self, local_scene_id, aliment_keyframe=False):
        
        alignment_trans = self.scene_info.alignment_trans
        
        local_scene_trans_s2t = alignment_trans['trans_s2t'][local_scene_id]
        local_scene_trans_s2t_R = alignment_trans['trans_s2t_R'][local_scene_id]
        
        # trans local to global  pose
        for cam in self.getTrainCameras()[local_scene_id]:
            cam.trans_pose(local_scene_trans_s2t, local_scene_trans_s2t_R)

        if aliment_keyframe:
            # aliment overlap keyframe
            overlap_pose_global = torch.inverse(self.getTrainCameras()[local_scene_id - 1][-1].world_view_transform.T) # c2w
            overlap_pose_local = torch.inverse(self.getTrainCameras()[local_scene_id][0].world_view_transform.T)
            
            # 
            trans_local_2_global = overlap_pose_global @ overlap_pose_local.inverse()
            trans_local_2_global_R = trans_local_2_global[:3, :3]
            for cam in self.getTrainCameras()[local_scene_id]:
                cam.trans_pose(trans_local_2_global, trans_local_2_global_R)

    def alignment_global_scene_pcd(self, local_scene_id, aliment_keyframe=True):
        '''
        首先： 姿态是场景中的姿态，也就是说，姿态发生了改变（从T1-> T2, 那么对应的点云也应该乘以相同的变换，原因是姿态的位置就是场景中的点云）
        明白了上述原理之后，那么
        '''
        alignment_trans = self.scene_info.alignment_trans
        
        local_scene_trans_s2t = alignment_trans['trans_s2t'][local_scene_id]
    
        # trans local to global pcd
        local_pcd = self.trans_BasicPointCloud(self.scene_info.point_cloud[local_scene_id])
        global_pcd = pcd_trans(local_pcd, local_scene_trans_s2t)

        if aliment_keyframe:
            # aliment overlap keyframe
            adjustment_pose = torch.inverse(self.getTrainCameras()[local_scene_id][-1].world_view_transform.T) # c2w
            org_pose = torch.inverse(self.getTrainCameras()[local_scene_id][-1].world_view_transform_origin.T)
            
            trans_aliment_keyframe = adjustment_pose @ org_pose.inverse()
            global_pcd = pcd_trans(global_pcd, trans_aliment_keyframe.cpu().numpy())
            
        self.prior_pcd = merge_pcd(self.prior_pcd, global_pcd)

        return global_pcd

    def debug_init_scene(self):
        '''
        srt用于将点云变换到全局场景
        '''
        alignment_trans = self.scene_info.alignment_trans
        trans_s2t = alignment_trans['trans_s2t']
        # trans_s2t_R = alignment_trans['trans_s2t_R']

        for i in range(len(trans_s2t)):
            if i == 0:
                pcd_g = self.trans_BasicPointCloud(self.scene_info.point_cloud[i])
                pcd_g = pcd_trans(pcd_g, trans_s2t[i])
            else:
                pcd_l = self.trans_BasicPointCloud(self.scene_info.point_cloud[i])
                pcd_l = pcd_trans(pcd_l, trans_s2t[i])
                pcd_g = merge_pcd(pcd_g, pcd_l)

        o3d.io.write_point_cloud(os.path.join(self.model_path, "debug_init.ply"), pcd_g)
        
    @staticmethod
    def trans_BasicPointCloud(pcd: BasicPointCloud):

        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd.points)
        pcd_o3d.colors = o3d.utility.Vector3dVector(pcd.colors)
        return pcd_o3d