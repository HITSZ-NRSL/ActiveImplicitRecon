import torch
import torch.nn as nn


def vec2skew(v):
    """
    :param v:  (3, ) torch tensor
    :return:   (3, 3)
    """
    zero = torch.zeros(1, dtype=torch.float32, device=v.device)
    skew_v0 = torch.cat([zero, -v[2:3], v[1:2]])  # (3, 1)
    skew_v1 = torch.cat([v[2:3], zero, -v[0:1]])
    skew_v2 = torch.cat([-v[1:2], v[0:1], zero])
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=0)  # (3, 3)
    return skew_v  # (3, 3)


def Exp(r):
    """so(3) vector to SO(3) matrix
    :param r: (3, ) axis-angle, torch tensor
    :return:  (3, 3)
    """
    skew_r = vec2skew(r)  # (3, 3)
    norm_r = r.norm() + 1e-15
    eye = torch.eye(3, dtype=torch.float32, device=r.device)
    R = (
        eye
        + (torch.sin(norm_r) / norm_r) * skew_r
        + ((1 - torch.cos(norm_r)) / norm_r**2) * (skew_r @ skew_r)
    )
    return R


def make_c2w(r, t):
    """
    :param r:  (3, ) axis-angle             torch tensor
    :param t:  (3, ) translation vector     torch tensor
    :return:   (4, 4)
    """
    R = Exp(r)  # (3, 3)
    c2w = torch.cat([R, t.unsqueeze(1)], dim=1)  # (3, 4)
    c2w = torch.cat(
        [c2w, torch.tensor([[0, 0, 0, 1]], dtype=c2w.dtype, device=c2w.device)], dim=0
    )  # (4, 4)
    return c2w


def quad2rotation(quad):
    """
    Convert quaternion to rotation in batch. Since all operation in pytorch, support gradient passing.

    Args:
        quad (tensor, 4): quaternion.

    Returns:
        rot_mat (tensor, 3*3): rotation.
    """
    qr, qi, qj, qk = quad[0], quad[1], quad[2], quad[3]
    two_s = 2.0 / (quad * quad).sum(-1)
    rot_mat = torch.zeros(3, 3).to(quad.device)
    rot_mat[0, 0] = 1 - two_s * (qj**2 + qk**2)
    rot_mat[0, 1] = two_s * (qi * qj - qk * qr)
    rot_mat[0, 2] = two_s * (qi * qk + qj * qr)
    rot_mat[1, 0] = two_s * (qi * qj + qk * qr)
    rot_mat[1, 1] = 1 - two_s * (qi**2 + qk**2)
    rot_mat[1, 2] = two_s * (qj * qk - qi * qr)
    rot_mat[2, 0] = two_s * (qi * qk - qj * qr)
    rot_mat[2, 1] = two_s * (qj * qk + qi * qr)
    rot_mat[2, 2] = 1 - two_s * (qi**2 + qj**2)

    return rot_mat


def get_camera_from_tensor(inputs):
    """
    Convert quaternion and translation to transformation matrix.

    """
    device = inputs.device
    quad, T = inputs[:4], inputs[4:]
    R = quad2rotation(quad)
    RT = torch.cat([R, T[:, None]], 1)
    homo = torch.tensor([[0, 0, 0, 1]], device=device)
    RT = torch.cat([RT, homo], 0)

    return RT


def position2polar(position):
    # r = distance of from the origin

    # phi = the reference angle from XY-plane (in a counter-clockwise direction from the x-axis)

    # theta = the reference angle from z-axis

    r = (position**2).sum().sqrt()
    if position[0] == 0 and position[1] == 0:
        phi = 0
    else:
        phi = torch.arctan2(position[1], position[0])
    theta = torch.arccos(position[2] / r)
    return torch.tensor([r, phi, theta], device=position.device)


def polar2heading_pose(polar):
    r, phi, theta = polar[0], polar[1], polar[2]
    x = r * torch.cos(phi) * torch.sin(theta)
    y = r * torch.sin(phi) * torch.sin(theta)
    z = r * torch.cos(theta)

    position = torch.tensor([x, y, z], device=polar.device)
    z_axis = -position / r
    if z_axis[0] == 0 and z_axis[1] == 0:
        x_axis = torch.tensor([1.0, 0.0, 0.0], device=polar.device)
    else:
        temp_z = torch.tensor([0.0, 0.0, 1.0], device=polar.device)
        x_axis = torch.cross(z_axis, temp_z)
    y_axis = torch.cross(z_axis, x_axis)

    pose = torch.zeros(4, 4, device=polar.device)
    pose[:3, 3] = position
    pose[:3, 2] = z_axis
    pose[:3, 1] = y_axis
    pose[:3, 0] = x_axis
    pose[3, 3] = 1
    return pose


def position2heading_pose(position):
    z_axis = -position / position.norm()
    if z_axis[0] == 0 and z_axis[1] == 0:
        y_axis = -torch.tensor([1.0, 0.0, 0.0], device=position.device)
    else:
        temp_z = torch.tensor([0.0, 0.0, 1.0], device=position.device)
        y_axis = -torch.cross(z_axis, temp_z)
    y_axis = y_axis/y_axis.norm()
    x_axis = torch.cross(y_axis,z_axis)

    pose = torch.zeros(4, 4, device=position.device)
    pose[:3, 3] = position
    pose[:3, 2] = z_axis
    pose[:3, 1] = y_axis
    pose[:3, 0] = x_axis
    pose[3, 3] = 1
    return pose


def position2tangent_basis(position):
    z_axis = -position / position.norm()
    if z_axis[0] == 0 and z_axis[1] == 0:
        x_axis = torch.tensor([1.0, 0.0, 0.0], device=position.device)
    else:
        temp_z = torch.tensor([0.0, 0.0, 1.0], device=position.device)
        x_axis = torch.cross(z_axis, temp_z)
    y_axis = torch.cross(z_axis, x_axis)

    return x_axis, y_axis, z_axis


class OdomPose(nn.Module):
    def __init__(self, init_c2w):
        """
        :param init_c2w: (4, 4) torch tensor
        """
        super(OdomPose, self).__init__()
        self.init_c2w = nn.Parameter(init_c2w, requires_grad=False)

        self.r = nn.Parameter(
            torch.tensor([1, 0, 0, 0], dtype=torch.float32), requires_grad=True
        )  # (4, )
        self.t = nn.Parameter(
            torch.zeros(size=(3,), dtype=torch.float32), requires_grad=True
        )  # (3, )

    def forward(self):
        r = self.r  # (3, ) axis-angle
        t = self.t  # (3, )
        # c2w = make_c2w(r, t)  # (4, 4)
        c2w = get_camera_from_tensor(torch.cat([r, t], 0))

        # learn a delta pose between init pose and target pose, if a init pose is provided
        c2w = c2w @ self.init_c2w

        return c2w


class GlobalPose(nn.Module):
    def __init__(self, num_cams, init_c2w, device):
        """
        :param num_cams:
        :param init_c2w: (N, 4, 4) torch tensor
        """
        super(GlobalPose, self).__init__()
        self.num_cams = num_cams

        self.init_c2w = nn.Parameter(init_c2w, requires_grad=False)
        self.device = device

        if self.num_cams > 1:
            quat = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32).repeat(
                num_cams - 1, 1
            )
            self.r = nn.Parameter(quat, requires_grad=True)  # (N, 4)
            self.t = nn.Parameter(
                torch.zeros(size=(num_cams - 1, 3), dtype=torch.float32),
                requires_grad=True,
            )  # (N, 3)

    def forward(self, cam_id):
        if cam_id == 0:
            c2w = torch.eye(4, device=self.device)
        else:
            r = self.r[cam_id - 1]  # (3, ) axis-angle
            t = self.t[cam_id - 1]  # (3, )
            # c2w = make_c2w(r, t)  # (4, 4)
            c2w = get_camera_from_tensor(torch.cat([r, t], 0))

        # learn a delta pose between init pose and target pose, if a init pose is provided
        c2w = c2w @ self.init_c2w[cam_id]

        return c2w

    def get_pose(self):
        with torch.no_grad():
            all_cam_pose = []
            for cam_num in range(self.num_cams):
                all_cam_pose.append(self(cam_num))
            all_cam_pose = torch.stack(all_cam_pose, 0)

        return all_cam_pose.detach()


class ViewPose(nn.Module):
    def __init__(self, init_c2w):
        """
        :param init_c2w: (3, 1) torch tensor
        """
        super(ViewPose, self).__init__()
        # self.init_view = nn.Parameter(position2polar(init_c2w), requires_grad=False)
        self.init_view = nn.Parameter(init_c2w, requires_grad=False)
        # self.x_axis, self.y_axis, self.z_axis = position2tangent_basis(init_c2w)
        self.pt = nn.Parameter(
            torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32), requires_grad=True
        )  # phi, theta

    def forward(self):
        nbv = self.init_view + self.pt

        # return polar2heading_pose(nbv)
        return position2heading_pose(nbv)

class SpherePose(nn.Module):
    def __init__(self, init_angle, radius, center, opt_radius=False):
        """
        :param init_angle: (2) torch tensor
        :param radius: float
        :param center: (3) torch tensor
        """
        super(SpherePose, self).__init__()
        self.center = nn.Parameter(center, requires_grad=False)
        self.radius = nn.Parameter(torch.tensor([radius]), requires_grad=opt_radius)
        self.m_z_world = nn.Parameter(torch.tensor([0.0, 0.0, -1.0]), requires_grad=False)
        self.homo = nn.Parameter(torch.tensor([[0.0, 0.0, 0.0, 1.0]]), requires_grad=False)
        
        self.angles = nn.Parameter(
            torch.tensor([init_angle[0], init_angle[1]], dtype=torch.float32), requires_grad=True
        )  # theta, phi in spherical coordinate system
        
    def forward(self):
        z = -torch.stack([torch.cos(self.angles[0]) * torch.cos(self.angles[1]), 
                          torch.cos(self.angles[0]) * torch.sin(self.angles[1]), 
                          torch.sin(self.angles[0])], 0)
        z = z/torch.norm(z)
        
        t = self.center - self.radius * z
        
        x = self.m_z_world - torch.dot(self.m_z_world, z) * z
        x = x / torch.norm(x)
        
        y = torch.cross(z, x)
        y = y/torch.norm(y)
        
        c2w = torch.stack([x, y, z, t], 1)
        c2w = torch.cat([c2w, self.homo], 0)
        
        return c2w
        