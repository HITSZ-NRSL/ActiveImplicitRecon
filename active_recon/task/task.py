import json
import os
import torch
import torch.multiprocessing as mp
from task.nbv import NBV
from task.sensor_rosnode import SensorROS

from task.vis_rosnode import VisROS

from model.network import HashNeRF, GridNeRF
from utils.misc import (
    generate_control_panel_img,
    load_views_pose,
    save_model,
    save_pose,
)

import cv2


class Task:
    def __init__(self, args):
        mp.set_sharing_strategy("file_system")
        mp.set_start_method("spawn", force=True)

        self.exp_path = args.exp_path
        if not os.path.exists(self.exp_path):
            os.makedirs(self.exp_path)
            os.makedirs(self.exp_path + "/color")
            os.makedirs(self.exp_path + "/depth")

        self.device = args.device
        self.frame_buffer_len = args.frame_buffer_len
        self.fx = args.fx
        self.fy = args.fy
        self.cx = args.cx
        self.cy = args.cy
        self.intrinsic = torch.tensor(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]],
            device=self.device,
        )
        self.height = args.height
        self.width = args.width

        # output intrinsic to camera_intrinsic.json
        with open(self.exp_path + "/camera_intrinsic.json", "w") as outfile:
            obj = json.dump(
                {
                    "width": self.width,
                    "height": self.height,
                    "intrinsic_matrix": [
                        self.fx,
                        0,
                        0,
                        0,
                        self.fy,
                        0,
                        self.cx,
                        self.cy,
                        1,
                    ],
                },
                outfile,
                indent=4,
            )
        with open(args.exp_path + "/tsdf.json", "w") as outfile:
            obj = json.dump(
                {
                    "path_dataset": args.exp_path + "/",
                    "path_intrinsic": args.exp_path + "/camera_intrinsic.json",
                    "depth_max": args.max_depth_range,
                    "tsdf_cubic_size": 1.0,
                    "debug_mode": False,
                },
                outfile,
                indent=4,
            )

        self.max_pixel_len = self.height * self.width
        self.pixel_buffer_len = int(
            self.max_pixel_len * args.pixel_buffer_portion)
        self.device = args.device

        if args.hash:
            self.network = HashNeRF(args.obj_size_guess).to(args.device)
        else:
            self.network = GridNeRF(args.obj_size_guess).to(args.device)

        self.g_kf_pose_list = torch.zeros(self.frame_buffer_len, 4, 4)
        self.g_kf_pose_list.share_memory_()
        self.g_gopt_kf_pose_list = torch.zeros(self.frame_buffer_len, 4, 4)
        self.g_gopt_kf_pose_list.share_memory_()
        self.g_kf_ray_list = torch.zeros(
            self.frame_buffer_len, self.pixel_buffer_len, 6
        )
        self.g_kf_ray_list.share_memory_()
        self.g_kf_color_list = torch.zeros(
            self.frame_buffer_len, self.pixel_buffer_len, 3
        )
        self.g_kf_color_list.share_memory_()
        self.g_kf_depth_list = torch.zeros(
            self.frame_buffer_len, self.pixel_buffer_len)
        self.g_kf_depth_list.share_memory_()
        self.g_kf_ray_len_list = torch.zeros(
            self.frame_buffer_len, dtype=torch.long)
        self.g_kf_ray_len_list.share_memory_()

        self.g_track_kf_idx = torch.zeros((1)).int()
        self.g_track_kf_idx.share_memory_()
        self.g_end_signal = torch.zeros(1).bool()
        self.g_end_signal.share_memory_()
        self.view_num = torch.zeros(1).long()
        self.view_num.share_memory_()

        self.views_pose = load_views_pose(args.views_pose_path).float()
        self.views_pose.share_memory_()
        self.views_gain = torch.zeros(1, self.views_pose.shape[0])
        self.views_gain.share_memory_()
        self.sample_views_pose = torch.zeros(200, 4, 4).float()
        self.sample_views_pose.share_memory_()
        self.sample_views_gain = torch.zeros(1, 200)
        self.sample_views_gain.share_memory_()

        self.vis_ros = VisROS(args, self)
        self.sensor_ros = SensorROS(args, self)
        self.nbv = NBV(args, self)

    def ctl_panel_run(self):
        cv2.imshow("Control panel", generate_control_panel_img())
        while True:
            key = cv2.waitKey()
            if key == 115 or key == 229:
                self.g_end_signal[0] = True
                print("Shutdown Control Panel")
                break

    def run(self, args):
        p_vis_ros = mp.Process(target=self.vis_ros.vis_run)
        if args.mode == 2:
            p_nbv = mp.Process(
                target=self.sensor_ros.offline_run, args=(args, self))
        elif args.mode == 1:
            p_nbv = mp.Process(
                target=self.sensor_ros.nbv_real_run, args=(args, self))
        else:
            p_nbv = mp.Process(
                target=self.sensor_ros.nbv_run, args=(args, self))

        print("Task: Start ROS Vis")
        p_vis_ros.start()
        print("Task: Start Candidate Views NBV")
        p_nbv.start()

        p_vis_ros.join()
        p_nbv.join()

        p_vis_ros.kill()
        p_nbv.kill()

    def save(self):
        path = self.exp_path
        model_path = os.path.join(path, "model.pt")
        save_model(model_path, self.network)

        kf_pose_path = os.path.join(path, "kf_pose.pkl")
        save_pose(kf_pose_path,
                  self.g_kf_pose_list[: self.g_track_kf_idx.item()])

        print("Model saved in:", model_path)
