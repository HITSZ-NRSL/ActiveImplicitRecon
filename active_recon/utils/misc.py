import numpy as np
import pickle
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

from model.network import HashNeRF, GridNeRF
import cv2


def save_pose(path, pose):
    with open(path, "wb") as f:
        pickle.dump(pose, f)


def load_pose(path):
    with open(path, "rb") as f:
        pose = pickle.load(f)

    return pose


def save_model(path, model):
    torch.save(model.state_dict(), path)


def load_model(path, hash, model_args):
    if hash:
        model = HashNeRF(**model_args)
    else:
        model = GridNeRF(**model_args)

    ckpt = torch.load(path)
    model.load_state_dict(ckpt)

    return model


def cal_relative_pose(obj_pose, cam_pose, device=torch.device("cuda")):
    if type(obj_pose) is np.ndarray:
        obj_pose = torch.from_numpy(obj_pose).to(device)
    if type(cam_pose) is np.ndarray:
        cam_pose = torch.from_numpy(cam_pose).to(device)

    cam2obj = torch.bmm(obj_pose.inverse(), cam_pose)

    return cam2obj


def freeze_net(net):
    for name, param in net.named_parameters():
        param.requires_grad = False


def unfreeze_net(net):
    for name, param in net.named_parameters():
        param.requires_grad = True


def dilate(x, kernel_size):
    if kernel_size % 2 == 0:
        raise ValueError("Expecting parameter kernel_size to be odd")

    padding = int((kernel_size - 1) / 2)
    res = F.max_pool2d(
        x, kernel_size=kernel_size, stride=1, padding=padding, dilation=1
    )

    return res


def erode(x, kernel_size):
    if kernel_size % 2 == 0:
        raise ValueError("Expecting parameter kernel_size to be odd")

    padding = int((kernel_size - 1) / 2)
    res = -F.max_pool2d(
        -x, kernel_size=kernel_size, stride=1, padding=padding, dilation=1
    )

    return res


def generate_control_panel_img():

    line_y = 30
    padding_x = 0
    padding_y = int(line_y * 0.7)
    res_image = np.zeros((30, 420, 3), np.uint8)
    cv2.putText(
        res_image,
        "Press 'S' or 's' to save model.",
        (padding_x, padding_y),
        cv2.FONT_HERSHEY_COMPLEX,
        0.8,
        (0, 255, 255),
        1,
        8,
        0,
    )
    return res_image


def load_views_pose(path):
    traj = np.loadtxt(path)
    traj_len = traj.shape[0]
    if traj.shape[1] == 8:
        traj_t = traj[:, 1:4]
        traj_q = traj[:, 4:]
    elif traj.shape[1] == 7:
        traj_t = traj[:, 0:3]
        traj_q = traj[:, 3:]

    rot = R.from_quat(traj_q)
    r_mat = R.as_matrix(rot)
    t_mat = np.expand_dims(traj_t, -1)
    homo = np.array([0, 0, 0, 1])
    homo = np.reshape(homo, (1, 1, 4))
    homo = np.repeat(homo, traj_len, 0)

    T = np.concatenate([r_mat, t_mat], axis=2)
    T = np.concatenate([T, homo], axis=1)

    return torch.from_numpy(T)


def disturb_pose(pose, disturb):
    rot = pose[:3, :3]
    rot_euler = R.from_matrix(rot).as_euler("zxy")
    trans = pose[:3, 3]

    rot_disturb = 2 * disturb[0] * (np.random.rand(3) - 0.5)
    trans_disturb = 2 * disturb[1] * (np.random.rand(3) - 0.5)

    rot_euler += rot_disturb
    rot = R.from_euler("zxy", rot_euler).as_matrix()
    trans += trans_disturb

    pose[:3, :3] = rot
    pose[:3, 3] = trans

    return pose


def rot_x(roll):
    return torch.tensor(
        [
            [1, 0, 0],
            [0, torch.cos(roll), -torch.sin(roll)],
            [0, torch.sin(roll), torch.cos(roll)],
        ]
    )


def rot_y(pitch):
    return torch.tensor(
        [
            [torch.cos(pitch), 0, -torch.sin(pitch)],
            [0, 1, 0],
            [torch.sin(pitch), 0, torch.cos(pitch)],
        ]
    )


def rot_z(yaw):
    return torch.tensor(
        [
            [torch.cos(yaw), -torch.sin(yaw), 0],
            [torch.sin(yaw), torch.cos(yaw), 0],
            [0, 0, 1],
        ]
    )
