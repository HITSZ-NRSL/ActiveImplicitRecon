from tkinter import Y
import torch

from utils.misc import rot_z


def cal_move_cost(cam_pose_start, cam_pose_set):
    cam_pose_start_norm = torch.norm(cam_pose_start[:3, 3])
    cam_pos_start_n = cam_pose_start[:3, 3] / cam_pose_start_norm
    cam_pose_set_norm = torch.norm(cam_pose_set[:3, 3])
    cam_pos_set_n = cam_pose_set[:3, 3] / cam_pose_set_norm

    omega = torch.arccos(torch.dot(cam_pos_start_n, cam_pos_set_n))
    return omega / torch.pi


class Planner:
    def __init__(self, args, task):
        self.device = args.device
        self.linear_max = args.linear_max

    def camera_pose_callback(self, cam_pose):
        self.cam_pose = cam_pose

    def reset(self, t_start, cam_pose_set):
        self.t_start = t_start
        self.cam_pose_start = self.cam_pose
        self.cam_pose_start_norm = torch.norm(self.cam_pose_start[:3, 3])
        self.cam_pos_start_n = self.cam_pose_start[:3, 3] / self.cam_pose_start_norm
        self.cam_pose_set = cam_pose_set
        self.cam_pose_set_norm = torch.norm(self.cam_pose_set[:3, 3])
        self.cam_pos_set_n = self.cam_pose_set[:3, 3] / self.cam_pose_set_norm

        self.angular = self.linear_max / torch.max(
            self.cam_pose_start_norm, self.cam_pose_set_norm
        )
        omega = torch.arccos(torch.dot(self.cam_pos_start_n, self.cam_pos_set_n))
        self.duration = omega / self.angular
        self.linear = (
            self.cam_pose_set_norm - self.cam_pose_start_norm
        ) / self.duration

        self.z_axis_W_CamStart_level = -self.cam_pos_start_n
        self.z_axis_W_CamStart_level[2] = 0
        self.z_axis_W_CamStart_level = (
            self.z_axis_W_CamStart_level / self.z_axis_W_CamStart_level.norm()
        )
        z_axis_W_CamSet_level = -self.cam_pos_set_n
        z_axis_W_CamSet_level[2] = 0
        z_axis_W_CamSet_level = z_axis_W_CamSet_level / z_axis_W_CamSet_level.norm()
        self.angular_yaw = (
            torch.arccos(torch.dot(self.z_axis_W_CamStart_level, z_axis_W_CamSet_level))
            / self.duration
        )

        # decouple to ensure camera always look down: [0, pi/2]
        z_axis_W_CamStart = self.cam_pose_start[:3, 2]
        self.pitch_start = torch.arcsin(z_axis_W_CamStart[2] / z_axis_W_CamStart.norm())
        z_axis_W_CamSet = self.cam_pose_set[:3, 2]
        pitch_set = torch.arcsin(z_axis_W_CamSet[2] / z_axis_W_CamSet.norm())
        self.angular_pitch = (pitch_set - self.pitch_start) / self.duration

        z_axis = torch.cross(self.cam_pos_start_n, self.cam_pos_set_n)
        if z_axis.norm() < 1e-6:
            z_axis = torch.tensor([0.0, 0.0, 1.0])
        elif z_axis[2] < 0:
            z_axis = -z_axis
            self.angular = -self.angular
            self.angular_yaw = -self.angular_yaw
        z_axis = z_axis / z_axis.norm()

        y_axis = torch.cross(z_axis, self.cam_pos_start_n)
        y_axis = y_axis / y_axis.norm()
        self.rot_W_CamStart = torch.stack([self.cam_pos_start_n, y_axis, z_axis], dim=1)

    def get_cam_pose_set(self, t):
        t = t - self.t_start
        if t > self.duration:
            t = self.duration
        norm = self.cam_pose_start_norm + self.linear * t
        theta = self.angular * t
        theta_yaw = self.angular_yaw * t
        pitch_set = self.pitch_start + self.angular_pitch * self.duration

        pos_W_Cam = (
            self.rot_W_CamStart
            @ torch.tensor([torch.cos(theta), torch.sin(theta), 0])
            * norm
        )
        z_axis_W_CamSet_level = rot_z(theta_yaw) @ self.z_axis_W_CamStart_level
        # x heading down
        x_axis_W_CamSet_level = torch.tensor([0.0, 0.0, -1.0])
        y_axis_W_CamSet_level = torch.cross(
            z_axis_W_CamSet_level, x_axis_W_CamSet_level
        )
        y_axis_W_CamSet_level = y_axis_W_CamSet_level / y_axis_W_CamSet_level.norm()
        rot_W_CamSet_level = torch.stack(
            [x_axis_W_CamSet_level, y_axis_W_CamSet_level, z_axis_W_CamSet_level], dim=1
        )

        pose_W_CamSet_level = torch.eye(4)
        pose_W_CamSet_level[:3, :3] = rot_W_CamSet_level
        pose_W_CamSet_level[:3, 3] = pos_W_Cam

        return pose_W_CamSet_level, pitch_set
