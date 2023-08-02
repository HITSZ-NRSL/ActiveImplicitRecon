import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils.geometry import sample_points_even_in_bbox
from utils.misc import cal_relative_pose

from .render import (
    render_color_depth_opacity_weight,
    render_occ_rgb,
    render_rays,
    transform_ray,
)


def vis_color_depth_img(
    model,
    pose,
    surf_points,
    obj_size_guess,
    width,
    height,
    intrinsic,
    max_rays_len,
    desc=None,
):
    device = pose.device
    point_u = torch.arange(0, width)
    point_v = torch.arange(0, height)
    point_u, point_v = torch.meshgrid(point_u, point_v, indexing="xy")
    point_u = point_u.reshape(-1).to(device)
    point_v = point_v.reshape(-1).to(device)

    point_x = (point_u - intrinsic[0][2]) / intrinsic[0][0]
    point_y = (point_v - intrinsic[1][2]) / intrinsic[1][1]
    point_z = torch.ones_like(point_x)

    rays_d = torch.stack([point_x, point_y, point_z], -1)

    rays_o = torch.zeros_like(rays_d)
    rays = torch.cat([rays_o, rays_d], -1)

    rays = transform_ray(rays, pose)

    rays, z_vals = sample_points_even_in_bbox(rays, surf_points, obj_size_guess)

    start_idx = 0
    while start_idx < rays.shape[0]:
        if start_idx + max_rays_len > rays.shape[0]:
            tmp_occs, tmp_rgbs = render_occ_rgb(
                model,
                rays[start_idx:],
                z_vals[start_idx:],
                surf_points,
            )
        else:
            tmp_occs, tmp_rgbs = render_occ_rgb(
                model,
                rays[start_idx : start_idx + max_rays_len],
                z_vals[start_idx : start_idx + max_rays_len],
                surf_points,
            )
        if start_idx == 0:
            occs = tmp_occs
            rgbs = tmp_rgbs
        else:
            occs = torch.cat([occs, tmp_occs], 0)
            rgbs = torch.cat([rgbs, tmp_rgbs], 0)
        start_idx += max_rays_len

    (
        render_colors,
        render_depths,
        _,
        _,
    ) = render_color_depth_opacity_weight(occs, rgbs, z_vals)

    rays = transform_ray(rays, pose.inverse())
    export_color_depth_img(
        rays, render_colors, render_depths, height, width, intrinsic, desc
    )


def export_color_depth_img(
    rays, render_colors, render_depths, height, width, intrinsic, desc=None
):
    point_x = rays[:, 3]
    point_y = rays[:, 4]
    point_u = point_x * intrinsic[0][0] + intrinsic[0][2]
    point_v = point_y * intrinsic[1][1] + intrinsic[1][2]
    point_u = point_u.round().long().cpu()
    point_v = point_v.round().long().cpu()
    depth_img = torch.zeros(height, width)
    depth_img[point_v, point_u] = render_depths.cpu()
    color_img = torch.zeros(3, height, width)
    color_img[:, point_v, point_u] = render_colors.permute(1, 0).to(torch.float32).cpu()

    depth_np = depth_img.numpy()
    color_np = color_img.numpy()
    fig, axs = plt.subplots(1, 2)
    fig.tight_layout()
    axs[0].imshow(depth_np, cmap="plasma", vmin=0, vmax=2)
    axs[0].set_title("Generated Depth")
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    color_np = np.clip(color_np, 0, 1)
    axs[1].imshow(color_np.swapaxes(0, 1).swapaxes(1, 2), cmap="plasma")
    axs[1].set_title("Generated RGB")
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    plt.subplots_adjust(wspace=0, hspace=0)
    if desc is None:
        desc = f"./render_color_depth.jpg"

    if not os.path.exists(os.path.dirname(desc)):
        os.makedirs(os.path.dirname(desc))

    plt.savefig(
        desc,
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.close()


def export_ig_img(
    rays, render_igs, height, width, intrinsic, exp_path, iter, down_size=2.0
):
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    max_ig = 0
    min_ig = 10000
    for igs in render_igs:
        tmp_max_ig = torch.max(igs).item()
        tmp_min_ig = torch.min(igs).item()
        if tmp_max_ig > max_ig:
            max_ig = tmp_max_ig
        if tmp_min_ig < min_ig:
            min_ig = tmp_min_ig
    gap_ig = max_ig - min_ig

    for view_num in range(len(rays)):
        point_x = rays[view_num][:, 3]
        point_y = rays[view_num][:, 4]
        point_u = point_x * intrinsic[0][0] + intrinsic[0][2]
        point_v = point_y * intrinsic[1][1] + intrinsic[1][2]
        point_u = (point_u / down_size).round().long().cpu()
        point_v = (point_v / down_size).round().long().cpu()
        ig_img = torch.zeros(
            (int)(height / down_size + 0.5), (int)(width / down_size + 0.5)
        )

        ig_img[point_v, point_u] = (render_igs[view_num] / gap_ig * 255).cpu()

        ig_np = ig_img.numpy().astype(np.uint8)
        ig_map_path = os.path.join(
            exp_path, str(iter) + "_" + str(view_num + 1) + "_ig.png"
        )
        ig_map = cv2.applyColorMap(ig_np, cv2.COLORMAP_PLASMA)
        cv2.imwrite(ig_map_path, ig_map)


def vis_gt_image(color_img, depth_img, desc):
    depth_np = depth_img.cpu().numpy()
    color_np = color_img.cpu().numpy()
    fig, axs = plt.subplots(1, 2)
    fig.tight_layout()
    depth_np[depth_np > 1.0] = 0.0
    print(depth_np.max())

    axs[0].imshow(depth_np, cmap="plasma", vmin=0, vmax=0.8)
    axs[0].set_title("GT Depth")
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    color_np = np.clip(color_np, 0, 1)
    axs[1].imshow(color_np)
    axs[1].set_title("GT RGB")
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    plt.subplots_adjust(wspace=0, hspace=0)
    if desc is None:
        desc = f"./gt_color_depth.jpg"

    if not os.path.exists(os.path.dirname(desc)):
        os.makedirs(os.path.dirname(desc))

    plt.savefig(
        desc,
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.close()


class Visualizer:
    """
    Visualize itermediate results, render out depth, color and depth uncertainty images.
    It can be called per iteration, which is good for debuging (to see how each tracking/mapping iteration performs).
    """

    def __init__(self, args):
        self.device = args.device
        self.target_id = args.target_id
        self.max_depth_range = args.max_depth_range
        self.exp_path = args.exp_path
        self.image_path = os.path.join(self.exp_path, "images")

        if not os.path.exists(self.image_path):
            os.makedirs(self.image_path)

        self.vis_iter_step = args.vis_iter_step
        self.vis_step = args.vis_step

        self.height = args.height
        self.width = args.width
        self.fx = args.fx
        self.fy = args.fy
        self.cx = args.cx
        self.cy = args.cy

        self.surf_points = args.surf_points
        self.strat_points = args.strat_points
        self.sigma = args.sigma

    def extract_rays(self, color, depth, mask, target_id, bbox=False):
        color = color[0].to(self.device)
        depth = depth[0][0].to(self.device)
        mask = mask[0][target_id].to(self.device)

        valid_depth_mask = (depth != 0.0) & (depth < self.max_depth_range)
        final_mask = mask & valid_depth_mask
        if bbox:
            point_mask = torch.nonzero(mask)
            mask_u_max = torch.max(point_mask[:, 1])
            mask_u_min = torch.min(point_mask[:, 1])
            mask_v_max = torch.max(point_mask[:, 0])
            mask_v_min = torch.min(point_mask[:, 0])
            final_mask[mask_v_min:mask_v_max, mask_u_min:mask_u_max] = 1

        point_image = torch.nonzero(final_mask)
        point_u = point_image[:, 1]
        point_v = point_image[:, 0]

        point_x = (point_u - self.cx) / self.fx
        point_y = (point_v - self.cy) / self.fy
        point_z = torch.ones_like(point_x)

        rays_d = torch.stack([point_x, point_y, point_z], -1)
        rays_o = torch.zeros_like(rays_d)
        rays = torch.cat([rays_o, rays_d], -1)

        colors = color[:, point_v, point_u].permute(1, 0)
        depths = depth[point_v, point_u]

        return rays, colors, depths

    def vis(self, model, data_num, pose, rays, depths, gt_color, gt_depth):
        """
        Visualization of depth, color images and save to file.
        """
        with torch.no_grad():
            # if data_num % self.vis_iter_step == 0:
            rays_transformed = transform_ray(rays, pose)
            render_colors, render_depths, _, _ = render_rays(
                model,
                rays_transformed,
                depths,
                self.surf_points,
                self.strat_points,
                self.sigma,
                self.cube_half_size
            )

            point_x = rays[:, 3]
            point_y = rays[:, 4]
            point_u = point_x * self.fx + self.cx
            point_v = point_y * self.fy + self.cy

            point_u = point_u.round().long()
            point_v = point_v.round().long()
            depth_img = torch.zeros_like(gt_depth.squeeze()).to(rays.device)
            depth_img[point_v, point_u] = render_depths
            color_img = torch.zeros_like(gt_color.squeeze()).to(rays.device)
            color_img[:, point_v, point_u] = render_colors.permute(1, 0).to(
                torch.float32
            )

            depth_np = depth_img.cpu().numpy()
            color_np = color_img.cpu().numpy()

            gt_depth_np = gt_depth.cpu().numpy()
            gt_color_np = gt_color.cpu().numpy()
            fig, axs = plt.subplots(2, 2)
            fig.tight_layout()
            max_depth = np.max(depth_np) * 1.5
            axs[0, 0].imshow(
                gt_depth_np.squeeze(), cmap="plasma", vmin=0, vmax=max_depth
            )
            axs[0, 0].set_title("Input Depth")
            axs[0, 0].set_xticks([])
            axs[0, 0].set_yticks([])
            axs[0, 1].imshow(depth_np, cmap="plasma", vmin=0, vmax=max_depth)
            axs[0, 1].set_title("Generated Depth")
            axs[0, 1].set_xticks([])
            axs[0, 1].set_yticks([])

            gt_color_np = np.clip(gt_color_np, 0, 1)
            color_np = np.clip(color_np, 0, 1)
            axs[1, 0].imshow(
                gt_color_np.squeeze().swapaxes(0, 1).swapaxes(1, 2), cmap="plasma"
            )
            axs[1, 0].set_title("Input RGB")
            axs[1, 0].set_xticks([])
            axs[1, 0].set_yticks([])
            axs[1, 1].imshow(color_np.swapaxes(0, 1).swapaxes(1, 2), cmap="plasma")
            axs[1, 1].set_title("Generated RGB")
            axs[1, 1].set_xticks([])
            axs[1, 1].set_yticks([])

            plt.subplots_adjust(wspace=0, hspace=0)
            plt.savefig(
                f"{self.image_path}/{data_num:05d}.jpg",
                bbox_inches="tight",
                pad_inches=0.2,
            )
            plt.close()

    def offline_vis(self, model):
        for data_num, data in enumerate(tqdm(self.dataloader)):
            if data_num % self.vis_step == 0:
                cam0_pose, obj_pose, mask, color, depth = data
                if mask[0][self.target_id].sum() < self.rays_per_img:
                    continue
                else:
                    rays, _, depths = self.extract_rays(
                        color, depth, mask, self.target_id
                    )
                gt_pose = (
                    cal_relative_pose(obj_pose[self.target_id], cam0_pose)
                    .to(self.device)
                    .squeeze()
                    .float()
                )
                self.vis(model, data_num, gt_pose, rays, depths, color, depth)
