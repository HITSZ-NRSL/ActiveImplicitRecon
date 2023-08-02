from pickle import NONE
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import apex.optimizers as optim
from task.planner import cal_move_cost
from utils.misc import (
    disturb_pose,
    freeze_net,
    load_pose,
    save_pose,
    unfreeze_net,
    load_views_pose,
    save_model,
)
from utils.geometry import register_pcd, sample_points_even_in_bbox

from utils.render import (
    render_occ_rgb,
    transform_ray,
    render_rays,
    render_void_rays,
    gen_random_rays,
    render_normal,
    render_normal,
)
from model.pose import GlobalPose, OdomPose, SpherePose, ViewPose
from model.network import HashNeRF, GridNeRF

import os
import open3d as o3d

from utils.visualizer import export_ig_img, vis_color_depth_img


def eval_rays_information_gain(occs):
    Io = -occs * torch.log(occs + 1e-6) - (1 - occs) * torch.log(1 - occs + 1e-6)
    Tv = torch.cumprod(1 - occs, dim=1)
    Iv = Tv * Io

    return Iv


def cal_angle_gain(visited_poses, pose):
    visited_vec = visited_poses[:, :3, 3].to(pose.device)
    pose_vec = pose[:3, 3]

    pose_vec_norm = pose_vec / torch.norm(pose_vec)
    visited_vec_norm = visited_vec / torch.norm(visited_vec, dim=-1).unsqueeze(-1)
    cos = visited_vec_norm @ pose_vec_norm

    return -torch.log(cos.max())


class NBV:
    def __init__(self, args, task):
        self.method = args.method
        self.exp_path = task.exp_path
        self.surf_points = args.surf_points
        self.strat_points = args.strat_points
        self.sigma = args.sigma
        self.cube_half_size = args.obj_size_guess
        self.depth_loss_lambda = args.depth_loss_lambda
        self.last_kf_num = 0
        self.rays_per_img = args.rays_per_img_recon
        self.active_kf_size = args.active_kf_size
        self.max_ray_len = self.rays_per_img * self.active_kf_size
        self.eval_rays = 12000

        self.max_depth_range = args.max_depth_range
        self.device = args.device
        self.verbose = args.verbose
        self.frame_buffer_len = args.frame_buffer_len

        self.init_realtime = False
        self.recon_step = args.recon_step
        self.final_step = 2000

        self.pose_opt = args.pose_opt
        self.global_pose_lr = args.global_pose_lr

        self.mode = args.mode

        self.rays_len = 0
        self.views_color = None
        self.views_depth = None
        self.views_ray_valid = None

        self.rays_void_len = 0
        self.views_ray_void = None

        self.pcd = None

        self.network_lr = args.network_lr
        self.views_gain = task.views_gain
        self.sample_views_pose = task.sample_views_pose
        self.sample_views_gain = task.sample_views_gain

        self.obj_size_guess = args.obj_size_guess

        self.width = args.width
        self.height = args.height

        self.intrinsic = task.intrinsic

        self.shared_network = task.network
        if args.hash:
            self.network = HashNeRF(args.obj_size_guess).to(args.device)
        else:
            self.network = GridNeRF(args.obj_size_guess).to(args.device)

        self.g_kf_pose_list = task.g_kf_pose_list
        self.g_kf_ray_list = task.g_kf_ray_list
        self.g_kf_color_list = task.g_kf_color_list
        self.g_kf_depth_list = task.g_kf_depth_list
        self.g_kf_ray_len_list = task.g_kf_ray_len_list

        self.view_num = task.view_num
        self.gopt_views_pose = task.g_gopt_kf_pose_list

        self.g_track_kf_idx = task.g_track_kf_idx
        self.g_end_signal = task.g_end_signal

        self.prev_recon_iter = 0
        self.track_frame_idx = 0

        self.vis_ros = task.vis_ros

        self.views_pose_path = args.views_pose_path
        self.disturb_pose = args.disturb_pose
        self.save_vis = args.save_vis

        # planning  # pure end2end
        self.last_nbv_angle = torch.tensor([0, torch.pi], device=self.device)

        # planning  # candidate views
        self.views_pose = load_views_pose(self.views_pose_path).float().to(self.device)
        self.visited_idx = [32]
        self.views_gain[0][32] = 0
        self.set_nbv_pose = None

    def extract_rays_rt_with_void(self, color, depth):
        color = color.to(self.device)
        depth = depth.to(self.device)
        device = color.device

        invalid_depth_mask = depth == 0.0
        outranged_depth_mask = depth > self.max_depth_range

        (v_max, u_max) = depth.shape

        u = torch.arange(u_max, device=device)
        v = torch.arange(v_max, device=device)
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        point_image = torch.stack([vv, uu], -1)

        point_image = point_image.view(-1, 2)
        invalid_depth_mask = invalid_depth_mask.view(-1)
        outranged_depth_mask = outranged_depth_mask.view(-1)

        point_u = point_image[:, 1]
        point_v = point_image[:, 0]

        point_x = (point_u - self.intrinsic[0][2]) / self.intrinsic[0][0]
        point_y = (point_v - self.intrinsic[1][2]) / self.intrinsic[1][1]
        point_z = torch.ones_like(point_x)

        rays_d = torch.stack([point_x, point_y, point_z], -1)
        rays_o = torch.zeros_like(rays_d)
        rays = torch.cat([rays_o, rays_d], -1)

        valid_depth_mask = ~invalid_depth_mask & ~outranged_depth_mask
        colors = color[point_v, point_u, :][valid_depth_mask]
        depths = depth[point_v, point_u][valid_depth_mask]

        rays_valid = rays[valid_depth_mask]
        rays_void = rays[outranged_depth_mask]

        return rays_valid, rays_void, colors, depths

    def global_optim(self, step):
        if self.pose_opt:
            global_pose = GlobalPose(
                self.view_num[0],
                self.gopt_views_pose[: self.view_num[0]],
                self.device,
            ).to(self.device)
            pose_optimizer = optim.FusedAdam(
                global_pose.parameters(), lr=self.global_pose_lr
            )
        else:

            def global_pose(ind):
                return self.gopt_views_pose[: self.view_num[0]][ind].to(self.device)

        network_optimizer = optim.FusedAdam(
            self.network.parameters(), lr=self.network_lr
        )

        max_ray_len_per_view = self.max_ray_len // self.view_num[0].item()

        for step_num in tqdm(range(step), desc="Recon: Global Optimization"):
            # for step_num in range(step):
            accu_rays = None
            accu_colors = None
            accu_depths = None
            accu_rays_void = None
            for i in range(self.view_num[0]):
                view_valid_start_idx = self.views_valid_start_idx[i]
                view_valid_len = self.views_valid_len[i]

                rays_ind = torch.randint(
                    view_valid_start_idx,
                    view_valid_start_idx + view_valid_len - 1,
                    (max_ray_len_per_view,),
                )
                view_ray_valid = transform_ray(
                    self.views_ray_valid[rays_ind], global_pose(i)
                )
                if accu_rays is None:
                    accu_rays = view_ray_valid
                    accu_colors = self.views_color[rays_ind]
                    accu_depths = self.views_depth[rays_ind]
                else:
                    accu_rays = torch.cat([accu_rays, view_ray_valid], 0)
                    accu_colors = torch.cat(
                        [accu_colors, self.views_color[rays_ind]], 0
                    )
                    accu_depths = torch.cat(
                        [accu_depths, self.views_depth[rays_ind]], 0
                    )

                view_void_start_idx = self.views_void_start_idx[i]
                view_void_len = self.views_void_len[i]

                if view_void_len > 0:
                    rays_void_ind = torch.randint(
                        view_void_start_idx,
                        view_void_start_idx + view_void_len - 1,
                        (max_ray_len_per_view,),
                    )

                    view_ray_void = transform_ray(
                        self.views_ray_void[rays_void_ind], global_pose(i).detach()
                    )
                    if accu_rays_void is None:
                        accu_rays_void = view_ray_void
                    else:
                        accu_rays_void = torch.cat([accu_rays_void, view_ray_void], 0)

            render_colors, render_depths, _, valid_mask, void_mask = render_rays(
                self.network,
                accu_rays,
                accu_depths,
                self.surf_points,
                self.strat_points,
                self.sigma,
                self.cube_half_size,
            )

            accu_colors = accu_colors[valid_mask]
            accu_depths = accu_depths[valid_mask]
            rays_void = accu_rays[void_mask]
            accu_rays = accu_rays[valid_mask]

            color_loss = F.mse_loss(render_colors, accu_colors)
            depth_loss = F.l1_loss(render_depths, accu_depths)

            loss = color_loss + self.depth_loss_lambda * depth_loss

            if accu_rays_void is not None:
                accu_rays_void = torch.cat([accu_rays_void, rays_void], 0)
                occ_void = render_void_rays(
                    self.network, accu_rays_void, self.surf_points, self.obj_size_guess
                )
                void_loss = -torch.log(1 - occ_void + 1e-6).mean()
                loss += 0.5 * void_loss

            if self.verbose:
                print(
                    f"Recon: Loss: {loss.item():05f}, Color Loss: {color_loss.item():05f}, Depth Loss: {depth_loss.item():05f}, Void Loss: {void_loss.item():05f}"
                )

            network_optimizer.zero_grad()
            if self.pose_opt:
                pose_optimizer.zero_grad()
            loss.backward()
            network_optimizer.step()
            if self.pose_opt:
                pose_optimizer.step()

            self.vis_ros.pub_render_pointcloud(accu_rays, render_colors, render_depths)

        if self.pose_opt:
            self.gopt_views_pose[: self.view_num[0]] = global_pose.get_pose()

    def eval_view_ig(self, tmp_spherepose):
        nbv_rays = gen_random_rays(
            self.width,
            self.height,
            self.eval_rays,
            self.intrinsic,
            self.device,
        )

        nbv_rays = transform_ray(nbv_rays, tmp_spherepose)

        nbv_rays, z_vals = sample_points_even_in_bbox(
            nbv_rays, self.surf_points, self.obj_size_guess
        )
        occs, _ = render_occ_rgb(self.network, nbv_rays, z_vals, self.surf_points)
        return self.eval_information_gain(occs)

    def run(self, color, depth, ref_pose):
        rays_valid, rays_void, colors, depths = self.extract_rays_rt_with_void(
            color, depth
        )

        if self.init_realtime == False:
            # common
            self.view_num[0] = 1
            self.gopt_views_pose[0] = ref_pose

            self.rays_len += rays_valid.shape[0]
            self.views_valid_start_idx = [0]
            self.views_valid_len = [rays_valid.shape[0]]
            self.views_ray_valid = rays_valid

            self.views_color = colors
            self.views_depth = depths

            self.rays_void_len += rays_void.shape[0]
            self.views_void_start_idx = [0]
            self.views_void_len = [rays_void.shape[0]]
            self.views_ray_void = rays_void

            self.pcd = o3d.geometry.PointCloud()
            self.disturb_pcd = o3d.geometry.PointCloud()
            self.init_realtime = True
        else:
            self.view_num[0] += 1

            if self.disturb_pose:
                ref_pose = torch.tensor(
                    disturb_pose(ref_pose.cpu().numpy(), [0.03, 0.05])
                )
            self.gopt_views_pose[self.view_num[0] - 1] = ref_pose

            self.views_valid_start_idx.append(self.rays_len)
            self.rays_len += rays_valid.shape[0]
            self.views_valid_len.append(rays_valid.shape[0])
            self.views_ray_valid = torch.cat([self.views_ray_valid, rays_valid], 0)

            self.views_color = torch.cat([self.views_color, colors], 0)
            self.views_depth = torch.cat([self.views_depth, depths], 0)

            self.views_void_start_idx.append(self.rays_void_len)
            self.rays_void_len += rays_void.shape[0]
            self.views_void_len.append(rays_void.shape[0])
            self.views_ray_void = torch.cat([self.views_ray_void, rays_void], 0)

        unfreeze_net(self.network)
        self.global_optim(self.recon_step)

        self.save_output(rays_valid, depths, ref_pose)

        if self.view_num[0].item() == 10 or self.g_end_signal[0]:
            self.global_optim(self.final_step)
            self.view_num[0] += 1
            self.save_output(rays_valid, depths, ref_pose)
            self.g_end_signal[0] = True
            self.shared_network.load_state_dict(self.network.state_dict())
            return
        else:
            if self.method == 0:
                # sample seed + end2end
                max_ig = 0
                with torch.no_grad():
                    for i in range(48):
                        sample_pose = self.sample_view_on_sphere()
                        tmp_sample_pose = sample_pose()
                        view_ig = self.eval_view_ig(sample_pose())
                        angle_gain = cal_angle_gain(
                            self.gopt_views_pose[: self.view_num],
                            tmp_sample_pose,
                        )
                        tmp_ig = view_ig + 1e-2 * angle_gain
                        if tmp_ig.isnan():
                            tmp_ig = 0

                        self.sample_views_gain[0][i] = tmp_ig
                        self.sample_views_pose[i] = tmp_sample_pose
                        if tmp_ig > max_ig:
                            max_ig = tmp_ig
                            nbv_pose = sample_pose

                pose_optimizer = optim.FusedAdam(nbv_pose.parameters(), lr=1e-2)
                freeze_net(self.network)

                for step in range(100):
                    tmp_spherepose = nbv_pose()
                    views_ig = self.eval_view_ig(tmp_spherepose)

                    # restrain the height of the camera to be positive
                    pose_loss = 1 - torch.sigmoid((tmp_spherepose[2, 3] - 0.1) / 1e-2)

                    loss = -views_ig + pose_loss

                    pose_optimizer.zero_grad()
                    loss.backward()
                    pose_optimizer.step()

                # next best view
                with torch.no_grad():
                    nbv_pose = nbv_pose()
            elif self.method == 1:
                # pure end2end
                spherepose = SpherePose(
                    self.last_nbv_angle, 0.6, torch.tensor([0.0, 0.0, 0.35]), True
                ).to(self.device)

                pose_optimizer = optim.FusedAdam(spherepose.parameters(), lr=1e-2)
                freeze_net(self.network)

                for step in range(100):
                    tmp_spherepose = spherepose()
                    views_ig = self.eval_view_ig(tmp_spherepose)

                    # restrain the height of the camera to be positive
                    pose_loss = 1 - torch.sigmoid((tmp_spherepose[2, 3] - 0.1) / 1e-2)

                    loss = -views_ig + pose_loss

                    pose_optimizer.zero_grad()
                    loss.backward()
                    pose_optimizer.step()

                # next best view
                with torch.no_grad():
                    nbv_pose = spherepose()
                    self.last_nbv_angle = spherepose.angles.clone().detach()
            elif self.method == 2:
                with torch.no_grad():
                    # information gain
                    self.views_ig = torch.zeros(self.views_pose.shape[0])
                    for idx in tqdm(
                        range(self.views_pose.shape[0]), desc="NBV: Searching"
                    ):
                        if idx in self.visited_idx:
                            continue

                        self.views_ig[idx] = self.eval_view_ig(self.views_pose[idx])

                    self.views_gain[0] = self.views_ig

                    # next best view
                    max_ig_idx = torch.max(self.views_gain[0], 0).indices.item()
                    nbv_pose = self.views_pose[max_ig_idx]
                    self.visited_idx.append(max_ig_idx)
            elif self.method == 3:
                with torch.no_grad():
                    nbv_pose = self.sample_view_on_sphere()()
            elif self.method == 4:
                with torch.no_grad():
                    nbv_pose = self.sample_view_in_shell(0.4)()
            elif self.method == 5:
                with torch.no_grad():
                    nbv_pose = self.circular_view(self.view_num[0].item(), 10)()
            elif self.method == 6:
                # new sample_seed + end2end
                max_ig = 0
                occ_thr = 0.1
                in_radius = 0.5

                with torch.no_grad():
                    for i in range(48):
                        sample_pose = self.sample_view_in_shell(in_radius)
                        while (
                            self.network(sample_pose()[:3, 3].view(1, 3))[0] > occ_thr
                        ):
                            sample_pose = self.sample_view_in_shell(in_radius)
                        tmp_sample_pose = sample_pose()
                        view_ig = self.eval_view_ig(sample_pose())
                        angle_gain = cal_angle_gain(
                            self.gopt_views_pose[: self.view_num],
                            tmp_sample_pose,
                        )
                        tmp_ig = view_ig + 1e-2 * angle_gain
                        if tmp_ig.isnan():
                            tmp_ig = 0

                        self.sample_views_gain[0][i] = tmp_ig
                        self.sample_views_pose[i] = tmp_sample_pose
                        if tmp_ig > max_ig:
                            max_ig = tmp_ig
                            nbv_pose = sample_pose

                pose_optimizer = optim.FusedAdam(nbv_pose.parameters(), lr=1e-2)
                freeze_net(self.network)

                for step in range(100):
                    tmp_spherepose = nbv_pose()
                    views_ig = self.eval_view_ig(tmp_spherepose)

                    # restrain the height of the camera to be positive
                    pose_loss = 1 - torch.sigmoid((tmp_spherepose[2, 3] - 0.1) / 1e-2)

                    # restrain the radius of the camera to away from object
                    if self.mode == 2:
                        pose_loss += 1 - torch.sigmoid(
                            (tmp_spherepose[:3, 3].norm() - in_radius) / 1e-2
                        )

                    collision_loss = torch.exp(
                        self.network(tmp_spherepose[:3, 3].view(1, 3))[0]
                    ) - torch.exp(torch.tensor(0.1))

                    loss = -views_ig + 10 * (pose_loss + collision_loss)

                    pose_optimizer.zero_grad()
                    loss.backward()
                    pose_optimizer.step()

                # next best view
                with torch.no_grad():
                    nbv_pose = nbv_pose()

            self.vis_ros.pub_nbv_pose(nbv_pose)
            return nbv_pose

    def sample_view_on_sphere(self, radius=0.6, height=0.05):
        sample_angles = torch.rand((2), device=self.device)
        sample_angles[0] = torch.asin(sample_angles[0])
        sample_angles[1] = 2 * torch.pi * sample_angles[1]

        sample_pose = SpherePose(
            sample_angles, radius, torch.tensor([0.0, 0.0, height])
        ).to(self.device)
        return sample_pose

    def sample_view_in_shell(self, in_radius=0.4, out_radius=0.6, height=0.05):
        sample_angles = torch.rand(2, device=self.device)
        sample_angles[0] = torch.asin(sample_angles[0])
        sample_angles[1] = 2 * torch.pi * sample_angles[1]
        sample_radius = (
            torch.rand(1, device=self.device) * (out_radius - in_radius) + in_radius
        )

        sample_pose = SpherePose(
            sample_angles, sample_radius, torch.tensor([0.0, 0.0, height]), True
        ).to(self.device)
        return sample_pose

    def circular_view(self, i, total):
        view_angles = torch.tensor([torch.pi / 6, 2 * torch.pi * i / total + torch.pi])

        sview_pose = SpherePose(view_angles, 0.6, torch.tensor([0.0, 0.0, 0.05])).to(
            self.device
        )
        return sview_pose

    def sample_views(self):
        with torch.no_grad():
            rand_view = 48
            max_ig = 0
            sample_angles = torch.rand((rand_view, 2), device=self.device)
            sample_angles = torch.stack(
                [
                    torch.asin(sample_angles[:, 0]),
                    2 * torch.pi * sample_angles[:, 1],
                ],
                1,
            )
            for seed_num in range(rand_view):
                sample_angle = sample_angles[seed_num]
                sample_pose = SpherePose(
                    sample_angle, 0.6, torch.tensor([0.0, 0.0, 0.0])
                ).to(self.device)
                tmp_sample_pose = sample_pose()

                view_ig = self.eval_view_ig(tmp_sample_pose)
                angle_gain = cal_angle_gain(
                    self.gopt_views_pose[: self.view_num],
                    tmp_sample_pose,
                )
                tmp_ig = view_ig + 1e-2 * angle_gain
                if tmp_ig.isnan():
                    tmp_ig = 0

                self.sample_views_gain[0][seed_num] = tmp_ig
                self.sample_views_pose[seed_num] = tmp_sample_pose
                if tmp_ig > max_ig:
                    max_ig = tmp_ig
                    nbv_angle = sample_angle
        return nbv_angle

    def save_output(self, rays_valid, depths, ref_pose):
        model_path = self.exp_path + "/" + str(self.view_num[0].item()) + ".pt"
        save_model(model_path, self.network)

        rays_valid_t = transform_ray(
            rays_valid,
            self.gopt_views_pose[self.view_num[0] - 1].clone().to(self.device),
        )
        pcd_path = os.path.join(
            self.exp_path, str(self.view_num[0].item()) + "single_pcd.ply"
        )
        single_pcd = o3d.geometry.PointCloud()
        single_pcd = register_pcd(single_pcd, rays_valid_t, depths, self.obj_size_guess)
        o3d.io.write_point_cloud(pcd_path, single_pcd)
        pcd_path = os.path.join(self.exp_path, str(self.view_num[0].item()) + "pcd.ply")
        self.pcd = register_pcd(self.pcd, rays_valid_t, depths, self.obj_size_guess)
        o3d.io.write_point_cloud(pcd_path, self.pcd)

        pose_path = os.path.join(
            self.exp_path, str(self.view_num[0].item()) + "_pose.pkl"
        )
        save_pose(pose_path, ref_pose)

        if self.save_vis:
            self.save_view_ig_pic(self.view_num[0].item() + 1)

            # for idx in range(self.views_pose.shape[0]):
            #     vis_color_depth_img(
            #         self.network,
            #         self.views_pose[idx],
            #         self.surf_points,
            #         self.obj_size_guess,
            #         self.width,
            #         self.height,
            #         self.intrinsic,
            #         self.max_ray_len,
            #         self.exp_path
            #         + "/nbv_views/"
            #         + str(self.view_num[0].item())
            #         + "_"
            #         + str(idx)
            #         + ".jpg",
            #     )

            # vis_gt_image(
            #     color,
            #     depth,
            #     self.exp_path + "/gt_image/" + str(self.visited_idx[-1]) + ".jpg",
            # )

            if self.disturb_pose:
                pcd_path = os.path.join(
                    self.exp_path,
                    str(self.view_num[0].item()) + "disturb_single_pcd.ply",
                )
                rays_valid_t = transform_ray(
                    rays_valid,
                    ref_pose.clone().to(self.device),
                )
                single_pcd = o3d.geometry.PointCloud()
                single_pcd = register_pcd(
                    single_pcd, rays_valid_t, depths, self.obj_size_guess
                )
                o3d.io.write_point_cloud(pcd_path, single_pcd)
                pcd_path = os.path.join(
                    self.exp_path, str(self.view_num[0].item()) + "disturb_pcd.ply"
                )
                self.disturb_pcd = register_pcd(
                    self.disturb_pcd, rays_valid_t, depths, self.obj_size_guess
                )
                o3d.io.write_point_cloud(pcd_path, self.disturb_pcd)

    def save_view_ig_pic(self, view_num, down_size=2.0):
        with torch.no_grad():
            device = self.device
            point_u = torch.arange(0, self.width - 1, down_size)
            point_v = torch.arange(0, self.height - 1, down_size)
            point_u, point_v = torch.meshgrid(point_u, point_v, indexing="xy")
            point_u = point_u.reshape(-1).to(device)
            point_v = point_v.reshape(-1).to(device)

            point_x = (point_u - self.intrinsic[0][2]) / self.intrinsic[0][0]
            point_y = (point_v - self.intrinsic[1][2]) / self.intrinsic[1][1]
            point_z = torch.ones_like(point_x)

            rays_d = torch.stack([point_x, point_y, point_z], -1)

            rays_o = torch.zeros_like(rays_d)
            rays = torch.cat([rays_o, rays_d], -1)

            views_igs = []
            views_rays = []
            views_max_igs = []
            views_max_rays = []
            for idx in range(self.views_pose.shape[0]):
                views_pose = self.views_pose[idx]
                tmp_rays = transform_ray(rays, views_pose)

                tmp_rays, z_vals = sample_points_even_in_bbox(
                    tmp_rays, 128, self.obj_size_guess
                )

                batch_ray_len = self.max_ray_len

                start_idx = 0
                while start_idx < tmp_rays.shape[0]:
                    if start_idx + batch_ray_len > tmp_rays.shape[0]:
                        tmp_occs, _ = render_occ_rgb(
                            self.network, tmp_rays[start_idx:], z_vals[start_idx:], 128
                        )

                    else:
                        tmp_occs, _ = render_occ_rgb(
                            self.network,
                            tmp_rays[start_idx : start_idx + batch_ray_len],
                            z_vals[start_idx : start_idx + batch_ray_len],
                            128,
                        )
                    if start_idx == 0:
                        occs = tmp_occs
                    else:
                        occs = torch.cat([occs, tmp_occs], 0)
                    start_idx += batch_ray_len

                tmp_igs = eval_rays_information_gain(occs)
                tmp_igs = tmp_igs.mean(1)

                tmp_rays = transform_ray(tmp_rays, views_pose.inverse())
                views_igs.append(tmp_igs)
                views_rays.append(tmp_rays)

                ray_num = (
                    (self.eval_rays * torch.cos(torch.tensor(view_num) / 20 * torch.pi))
                    .ceil()
                    .int()
                )
                tmp_igs, tmp_indices = torch.sort(tmp_igs)
                views_max_igs.append(tmp_igs[-ray_num:])
                views_max_rays.append(tmp_rays[tmp_indices[-ray_num:]])
            export_ig_img(
                views_rays,
                views_igs,
                self.height,
                self.width,
                self.intrinsic,
                self.exp_path + "/ig_all",
                view_num,
                down_size,
            )
            export_ig_img(
                views_max_rays,
                views_max_igs,
                self.height,
                self.width,
                self.intrinsic,
                self.exp_path + "/ig_max",
                view_num,
                down_size,
            )

    def eval_max_rays_information_gain(self, occs):
        ig = eval_rays_information_gain(occs)

        ig = ig.mean(1)

        ray_num = (
            (
                self.eval_rays
                * torch.cos(torch.tensor(self.view_num[0].item()) / 30 * torch.pi)
            )
            .ceil()
            .int()
        )
        return torch.sort(ig).values[-ray_num:], torch.sort(ig).indices[-ray_num:]

    def eval_information_gain(self, occs):
        igs, _ = self.eval_max_rays_information_gain(occs)
        return igs.mean()
