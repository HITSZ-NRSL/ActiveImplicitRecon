import rospy
from geometry_msgs.msg import PoseStamped, Pose, Point
from nav_msgs.msg import Path
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import PointCloud2, PointField

import torch
import numpy as np

from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


def tensor_to_pose(tensor_pose):
    pose = Pose()
    pose.position.x = tensor_pose[0, 3]
    pose.position.y = tensor_pose[1, 3]
    pose.position.z = tensor_pose[2, 3]

    quat = R.from_matrix(tensor_pose[:3, :3]).as_quat()
    pose.orientation.w = quat[3]
    pose.orientation.x = quat[0]
    pose.orientation.y = quat[1]
    pose.orientation.z = quat[2]

    # if (
    #     tensor_pose[:3, 3].norm() < 0.1
    #     or tensor_pose[:3, 3].norm() > 5
    #     or tensor_pose[:3, :3].det() < 0.9
    # ):
    #     print("Warning: pose is out of safe area")
    #     print(tensor_pose)
    #     return None

    return pose


def pose_to_cam_marker(pose, color_rgba=ColorRGBA(1, 0, 0, 1), scale=0.2, id=0):

    marker = Marker()
    marker.ns = str(id)
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = scale / 10.0

    tmp_pose = tensor_to_pose(pose)
    if tmp_pose is None:
        return None
    marker.pose = tmp_pose

    cam_model = torch.Tensor(
        [
            [-1.0, -0.5, 1.0],  # lt
            [1.0, -0.5, 1.0],  # rt
            [-1.0, 0.5, 1.0],  # lb
            [1.0, 0.5, 1.0],  # rb
            [-0.7, -0.5, 1.0],  # lt0
            [-0.7, -0.2, 1.0],  # lt1
            [-1.0, -0.2, 1.0],  # lt2
            [0.0, 0.0, 0.0],
        ]
    )
    cam_model = cam_model * scale

    # image boundaries
    marker.points.append(Point(cam_model[0, 0], cam_model[0, 1], cam_model[0, 2]))
    marker.points.append(Point(cam_model[2, 0], cam_model[2, 1], cam_model[2, 2]))
    marker.colors.append(color_rgba)
    marker.colors.append(color_rgba)

    marker.points.append(Point(cam_model[2, 0], cam_model[2, 1], cam_model[2, 2]))
    marker.points.append(Point(cam_model[3, 0], cam_model[3, 1], cam_model[3, 2]))
    marker.colors.append(color_rgba)
    marker.colors.append(color_rgba)

    marker.points.append(Point(cam_model[3, 0], cam_model[3, 1], cam_model[3, 2]))
    marker.points.append(Point(cam_model[1, 0], cam_model[1, 1], cam_model[1, 2]))
    marker.colors.append(color_rgba)
    marker.colors.append(color_rgba)

    marker.points.append(Point(cam_model[1, 0], cam_model[1, 1], cam_model[1, 2]))
    marker.points.append(Point(cam_model[0, 0], cam_model[0, 1], cam_model[0, 2]))
    marker.colors.append(color_rgba)
    marker.colors.append(color_rgba)

    # top-left indicator
    marker.points.append(Point(cam_model[4, 0], cam_model[4, 1], cam_model[4, 2]))
    marker.points.append(Point(cam_model[5, 0], cam_model[5, 1], cam_model[5, 2]))
    marker.colors.append(color_rgba)
    marker.colors.append(color_rgba)

    marker.points.append(Point(cam_model[5, 0], cam_model[5, 1], cam_model[5, 2]))
    marker.points.append(Point(cam_model[6, 0], cam_model[6, 1], cam_model[6, 2]))
    marker.colors.append(color_rgba)
    marker.colors.append(color_rgba)

    # optical center connector
    marker.points.append(Point(cam_model[0, 0], cam_model[0, 1], cam_model[0, 2]))
    marker.points.append(Point(cam_model[7, 0], cam_model[7, 1], cam_model[7, 2]))
    marker.colors.append(color_rgba)
    marker.colors.append(color_rgba)

    marker.points.append(Point(cam_model[1, 0], cam_model[1, 1], cam_model[1, 2]))
    marker.points.append(Point(cam_model[7, 0], cam_model[7, 1], cam_model[7, 2]))
    marker.colors.append(color_rgba)
    marker.colors.append(color_rgba)

    marker.points.append(Point(cam_model[2, 0], cam_model[2, 1], cam_model[2, 2]))
    marker.points.append(Point(cam_model[7, 0], cam_model[7, 1], cam_model[7, 2]))
    marker.colors.append(color_rgba)
    marker.colors.append(color_rgba)

    marker.points.append(Point(cam_model[3, 0], cam_model[3, 1], cam_model[3, 2]))
    marker.points.append(Point(cam_model[7, 0], cam_model[7, 1], cam_model[7, 2]))
    marker.colors.append(color_rgba)
    marker.colors.append(color_rgba)

    return marker


class VisROS:
    def __init__(self, args, task):
        self.pub_pose_signal = torch.zeros((1)).int()
        self.pub_pose_signal.share_memory_()
        self.g_pose_gt = torch.zeros(1, 4, 4)
        self.g_pose_gt.share_memory_()
        self.pose_nbv = torch.zeros(1, 4, 4)
        self.pose_nbv.share_memory_()

        self.xyz_rgb_len = torch.zeros((1)).int()
        self.xyz_rgb_len.share_memory_()
        self.xyz_rgb_list = torch.zeros(
            1, args.frame_buffer_len * args.rays_per_img_recon, 6
        )
        self.xyz_rgb_list.share_memory_()

        self.xyz_d_len = torch.zeros((1)).int()
        self.xyz_d_len.share_memory_()
        self.xyz_d_list = torch.zeros(
            1, args.frame_buffer_len * args.rays_per_img_recon, 4
        )
        self.xyz_d_list.share_memory_()

        self.opt_kf_idx = (
            torch.ones(1, args.frame_buffer_len) * args.frame_buffer_len
        ).int()
        self.opt_kf_idx.share_memory_()

        self.g_end_signal = task.g_end_signal

        self.g_gopt_kf_pose_list = task.g_gopt_kf_pose_list
        self.view_num = task.view_num

        self.rays_per_img_recon = args.rays_per_img_recon
        self.frame_buffer_len = args.frame_buffer_len

        self.views_pose = task.views_pose
        self.views_gain = task.views_gain
        self.sample_views_pose = task.sample_views_pose
        self.sample_views_gain = task.sample_views_gain

    def get_opt_pointcloud(self, header):
        xyzrgb = self.xyz_rgb_list[0, : self.xyz_rgb_len[0]]
        xyz, rgb = xyzrgb[:, :3], xyzrgb[:, 3:]

        xyz_np = xyz.numpy()
        rgb_np = np.zeros((xyz_np.shape[0], 4), dtype=np.uint8) + 255
        rgb_np[:, :3] = (rgb * 255).numpy().astype(np.uint8)
        rgb_np[:, [0, 2]] = rgb_np[:, [2, 0]]

        pts_rgb = np.zeros(
            (xyz_np.shape[0], 1),
            dtype={
                "names": ("x", "y", "z", "rgba"),
                "formats": ("f4", "f4", "f4", "u4"),
            },
        )
        pts_rgb["x"] = xyz_np[:, 0].reshape((-1, 1))
        pts_rgb["y"] = xyz_np[:, 1].reshape((-1, 1))
        pts_rgb["z"] = xyz_np[:, 2].reshape((-1, 1))
        pts_rgb["rgba"] = rgb_np.view("uint32")

        pcl_msg = PointCloud2()
        pcl_msg.header = header
        pcl_msg.height = 1
        pcl_msg.width = xyz_np.shape[0]
        pcl_msg.fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            # PointField("rgb", 12, PointField.UINT32, 1),
            PointField("rgba", 12, PointField.UINT32, 1),
        ]
        pcl_msg.is_bigendian = False
        pcl_msg.point_step = 16
        pcl_msg.row_step = pcl_msg.point_step * pcl_msg.width
        pcl_msg.is_dense = int(np.isfinite(xyz_np).all())
        pcl_msg.data = pts_rgb.tostring()
        return pcl_msg

    def get_views_markers(self, header):
        views_mks = MarkerArray()
        view_nonzero = torch.nonzero(self.views_gain[0])
        min_ig = 0.0
        gap_ig = 1.0
        if view_nonzero.shape[0] > 0:
            max_ig = torch.max(self.views_gain[0][view_nonzero]).item()
            min_ig = torch.min(self.views_gain[0][view_nonzero]).item()
            gap_ig = max_ig - min_ig
            if gap_ig == 0:
                gap_ig = max_ig
        for idx in range(self.views_pose.shape[0]):
            if self.views_gain[0][idx].item() == 0:
                view_mk = pose_to_cam_marker(
                    self.views_pose[idx], ColorRGBA(1, 0, 0, 0), 0.1, idx
                )
            else:
                rgba = plt.cm.rainbow(
                    (self.views_gain[0][idx].item() - min_ig) / gap_ig
                )
                view_mk = pose_to_cam_marker(
                    self.views_pose[idx],
                    ColorRGBA(rgba[0], rgba[1], rgba[2], rgba[3]),
                    0.06,
                    idx,
                )
            view_mk.header = header
            views_mks.markers.append(view_mk)
        return views_mks

    def get_sample_views_markers(self, header):
        views_mks = MarkerArray()
        view_nonzero = torch.nonzero(self.sample_views_gain[0])
        min_ig = 0.0
        gap_ig = 1.0
        if view_nonzero.shape[0] > 0:
            max_ig = torch.max(self.sample_views_gain[0][view_nonzero]).item()
            min_ig = torch.min(self.sample_views_gain[0][view_nonzero]).item()
            gap_ig = max_ig - min_ig
            if gap_ig == 0:
                gap_ig = max_ig
        for idx in range(self.sample_views_pose.shape[0]):
            if self.sample_views_pose[idx][3, 3] == 1:
                if self.sample_views_gain[0][idx].item() == 0:
                    view_mk = pose_to_cam_marker(
                        self.sample_views_pose[idx], ColorRGBA(1, 0, 0, 0), 0.1, idx
                    )
                else:
                    rgba = plt.cm.rainbow(
                        (self.sample_views_gain[0][idx].item() - min_ig) / gap_ig
                    )
                    view_mk = pose_to_cam_marker(
                        self.sample_views_pose[idx],
                        ColorRGBA(rgba[0], rgba[1], rgba[2], rgba[3]),
                        0.06,
                        idx,
                    )
                view_mk.header = header
                views_mks.markers.append(view_mk)
        return views_mks

    def vis_run(self):
        rospy.init_node("active_recon_vis")
        rate = rospy.Rate(10)

        pub_pose_gt = rospy.Publisher("pose_gt", PoseStamped, queue_size=10)
        pub_path_gt = rospy.Publisher("path_gt", Path, queue_size=10)
        pub_cam_vis_gt = rospy.Publisher("cam_vis_gt", MarkerArray, queue_size=10)
        pub_cam_vis_nbv = rospy.Publisher("cam_vis_nbv", MarkerArray, queue_size=10)

        pub_kfs_vis = rospy.Publisher("keyframes_vis_test", MarkerArray, queue_size=10)
        pub_kfs_pc = rospy.Publisher("keyframes_point_cloud", PointCloud2, queue_size=2)

        pub_cam_vis_nbv_igs = rospy.Publisher(
            "cam_vis_nbv_igs", MarkerArray, queue_size=10
        )
        pub_cam_vis_sample_nbv_igs = rospy.Publisher(
            "cam_vis_sample_nbv_igs", MarkerArray, queue_size=10
        )

        path_gt = Path()

        while not rospy.is_shutdown():

            header = Header(stamp=rospy.Time.now(), frame_id="map")
            if self.g_pose_gt[0][3, 3] == 1:
                g_pose_gt = self.g_pose_gt[0].clone()
                pose_gt_msg = PoseStamped()
                pose_gt_msg.header = header
                tmp_pose = tensor_to_pose(g_pose_gt)
                if tmp_pose is not None:
                    pose_gt_msg.pose = tmp_pose
                    pub_pose_gt.publish(pose_gt_msg)

                    path_gt.header = header
                    path_gt.poses.append(pose_gt_msg)
                    pub_path_gt.publish(path_gt)

                    cam_vis_gt = MarkerArray()
                    cam_marker_gt = pose_to_cam_marker(g_pose_gt, ColorRGBA(0, 1, 0, 1))
                    cam_marker_gt.header = header
                    cam_vis_gt.markers.append(cam_marker_gt)
                    pub_cam_vis_gt.publish(cam_vis_gt)

            if self.pose_nbv[0][3, 3] == 1:
                cam_vis_nbv = MarkerArray()
                cam_marker_nbv = pose_to_cam_marker(
                    self.pose_nbv[0], ColorRGBA(1, 1, 0, 1)
                )
                cam_marker_nbv.header = header
                cam_vis_nbv.markers.append(cam_marker_nbv)
                pub_cam_vis_nbv.publish(cam_vis_nbv)

            gopt_kf_len = self.view_num.clone().item()
            if gopt_kf_len > 0:
                kfs_vis = MarkerArray()
                if gopt_kf_len > self.frame_buffer_len:
                    gopt_kf_len = self.frame_buffer_len
                for kf_id in range(gopt_kf_len):
                    g_pose_est = self.g_gopt_kf_pose_list[kf_id].clone()
                    if (self.opt_kf_idx[0] == kf_id).sum() > 0:
                        kf_mk = pose_to_cam_marker(
                            g_pose_est, ColorRGBA(1, 0.5, 0, 1), 0.1, kf_id
                        )
                    else:
                        kf_mk = pose_to_cam_marker(
                            g_pose_est, ColorRGBA(1, 1, 0, 1), 0.08, kf_id
                        )
                    if kf_mk is not None:
                        kf_mk.header = header
                        kfs_vis.markers.append(kf_mk)
                pub_kfs_vis.publish(kfs_vis)

            pub_kfs_pc.publish(self.get_opt_pointcloud(header))

            views_mks = self.get_views_markers(header)
            pub_cam_vis_nbv_igs.publish(views_mks)

            sample_views_mks = self.get_sample_views_markers(header)
            pub_cam_vis_sample_nbv_igs.publish(sample_views_mks)

            if self.g_end_signal:
                print("Shutdown VisROS")
                break
            rate.sleep()

    def pub_pose_gt(self, pose_gt):
        self.g_pose_gt[0] = pose_gt

    def pub_nbv_pose(self, nbv_pose):
        self.pose_nbv[0] = nbv_pose

    def pub_render_pointcloud(self, accu_rays, render_colors, render_depths):
        rays_o, rays_d = accu_rays[:, :3], accu_rays[:, 3:6]
        xyz = rays_o + rays_d * render_depths.unsqueeze(-1)
        self.xyz_rgb_len[0] = xyz.shape[0]
        self.xyz_rgb_list[0, : self.xyz_rgb_len[0]] = torch.cat(
            [xyz, render_colors], -1
        ).detach()
