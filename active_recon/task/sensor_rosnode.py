import os
import cv2
import rospy
from task.nbv import NBV
from task.planner import Planner
from task.vis_rosnode import tensor_to_pose
from gazebo_msgs.srv import (
    GetModelState,
    GetModelStateRequest,
    SetModelStateRequest,
    SetModelState,
)
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import message_filters
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from utils.misc import rot_y
from utils.tagdet import TagDetector, create_board, create_box, initDetector


def model_state_to_torch_pose(model_state, device="cpu"):
    torch_pose = torch.zeros(4, 4, device=device)
    torch_pose[0, 3] = model_state.pose.position.x
    torch_pose[1, 3] = model_state.pose.position.y
    torch_pose[2, 3] = model_state.pose.position.z
    torch_pose[:3, :3] = torch.from_numpy(
        R.from_quat(
            [
                model_state.pose.orientation.x,
                model_state.pose.orientation.y,
                model_state.pose.orientation.z,
                model_state.pose.orientation.w,
            ]
        ).as_matrix()
    ).to(device)

    torch_pose[3, 3] = 1
    return torch_pose


def ros2tensorpose(ros_pose):
    tensor_pose = torch.eye(4)
    tensor_pose[0, 3] = ros_pose.position.x
    tensor_pose[1, 3] = ros_pose.position.y
    tensor_pose[2, 3] = ros_pose.position.z

    tensor_pose[:3, :3] = torch.from_numpy(
        R.from_quat(
            [
                ros_pose.orientation.x,
                ros_pose.orientation.y,
                ros_pose.orientation.z,
                ros_pose.orientation.w,
            ]
        ).as_matrix()
    )

    return tensor_pose


def tensor2rospose(tensor_pose):
    ros_pose = Pose()
    ros_pose.position.x = tensor_pose[0, 3]
    ros_pose.position.y = tensor_pose[1, 3]
    ros_pose.position.z = tensor_pose[2, 3]

    quat = R.from_matrix(tensor_pose[:3, :3]).as_quat()
    ros_pose.orientation.w = quat[3]
    ros_pose.orientation.x = quat[0]
    ros_pose.orientation.y = quat[1]
    ros_pose.orientation.z = quat[2]

    return ros_pose


def check_tensorpose(tensor_pose):
    if (
        tensor_pose[2, 3] < 0.0
        or tensor_pose[:3, 3].norm() < 0.1
        or tensor_pose[:3, 3].norm() > 5
        or tensor_pose[:3, :3].det() < 0.9
        or torch.any(torch.isnan(tensor_pose))
    ):
        print("Warning: pose is out of safe area")
        print(tensor_pose)
        return False
    else:
        return True


class SensorROS:
    def __init__(self, args, task):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.method = args.method
        self.intrinsic = task.intrinsic.cpu().numpy()
        self.g_end_signal = task.g_end_signal
        self.vis_ros = task.vis_ros

        self.view_num = task.view_num
        self.exp_path = args.exp_path

        self.get_model_state_rqt = GetModelStateRequest()
        self.get_model_state_rqt.model_name = "d435_model"
        self.set_model_state_rqt = SetModelStateRequest()
        self.set_model_state_rqt.model_state.model_name = "d435_model"
        self.set_model_state_rqt.model_state.reference_frame = "world"

        self.cam_to_body = torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )

        self.body_to_cam = torch.tensor(
            [
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )

        self.keyframe_num = 0
        self.keyframe_num_last = 0

        self.key_poses = []
        self.set_nbv_pose = None

    def cam_callback(self, color_msg, depth_msg):
        if (
            rospy.Time.now().to_sec() - color_msg.header.stamp.to_sec() < 0.01
            and color_msg.header.stamp.to_sec() - self.set_pose_time > 0.1
        ):
            cam_state = self.get_state_service(self.get_model_state_rqt)

            if cam_state.success:
                cam_pose = (
                    model_state_to_torch_pose(cam_state, device=self.device)
                    @ self.cam_to_body
                )

                # save cam_pose as kitti format
                self.key_poses.append(cam_pose.cpu().numpy().flatten())
                np.savetxt(self.exp_path + "/poses.txt", self.key_poses)

                bridge = CvBridge()
                color_image = bridge.imgmsg_to_cv2(color_msg, "bgr8")
                cv2.imwrite(
                    "%s/%06d.png" % (self.exp_path + "/color",
                                     self.view_num[0].item()),
                    color_image,
                )
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                color_image = color_image / 255.0
                color_image = torch.from_numpy(
                    color_image).float().to(self.device)

                depth_image = bridge.imgmsg_to_cv2(depth_msg, "16UC1")
                cv2.imwrite(
                    "%s/%06d.png" % (self.exp_path + "/depth",
                                     self.view_num[0].item()),
                    depth_image,
                )
                depth_image = depth_image.astype(np.float32) / 1000.0
                depth_image = torch.from_numpy(depth_image).to(self.device)

                self.set_nbv_pose = self.nbv.run(
                    color_image,
                    depth_image,
                    cam_pose,
                )

                if self.set_nbv_pose is not None:
                    self.set_nbv_pose = self.set_nbv_pose.cpu()

                    pose_gt = (
                        model_state_to_torch_pose(
                            self.get_state_service(self.get_model_state_rqt)
                        )
                        @ self.cam_to_body.cpu()
                    )
                    self.vis_ros.pub_pose_gt(pose_gt)

                    self.planner.camera_pose_callback(pose_gt)
                    self.planner.reset(
                        rospy.Time.now().to_sec(), self.set_nbv_pose)

                    while (pose_gt - self.set_nbv_pose).abs().sum() > 0.01:
                        pose_gt = (
                            model_state_to_torch_pose(
                                self.get_state_service(
                                    self.get_model_state_rqt)
                            )
                            @ self.cam_to_body.cpu()
                        )
                        self.vis_ros.pub_pose_gt(pose_gt)

                        pose_W_CamSet_level, pitch = self.planner.get_cam_pose_set(
                            rospy.Time.now().to_sec()
                        )

                        pose_Level_Set = torch.eye(4)
                        pose_Level_Set[:3, :3] = rot_y(pitch)

                        body_pose_set = tensor_to_pose(
                            pose_W_CamSet_level
                            @ pose_Level_Set
                            @ self.body_to_cam.cpu()
                        )
                        if body_pose_set is not None:
                            self.set_model_state_rqt.model_state.pose = body_pose_set
                            self.set_state_service(self.set_model_state_rqt)

                        else:
                            raise Exception("Set Pose is out of safe area!")

                    self.set_pose_time = rospy.Time.now().to_sec()

        if self.g_end_signal:
            print("Shutdown SensorROS")
            rospy.signal_shutdown("shutdown")

    def real_callback(self, color_msg, depth_msg):
        if rospy.Time.now().to_sec() - color_msg.header.stamp.to_sec() < 0.5:
            # Image
            bridge = CvBridge()
            # CompressedImage
            color_arr = np.fromstring(color_msg.data, np.uint8)
            color_image = cv2.imdecode(color_arr, cv2.IMREAD_COLOR)
            id_len, cam_pose = TagDetector(
                self.detector, color_image, self.board, self.intrinsic
            )
            if id_len > 0:
                cam_pose = torch.from_numpy(cam_pose).float()
                self.vis_ros.pub_pose_gt(cam_pose)
                recon_signal = False
                if self.view_num[0].item() == 0:
                    recon_signal = True
                if self.set_nbv_pose is not None:
                    z_cam_pose = cam_pose[:3, 2].clone()
                    z_cam_pose[2] = 0
                    z_cam_pose = z_cam_pose / z_cam_pose.norm()
                    z_set_nbv_pose = self.set_nbv_pose[:3, 2].clone()
                    z_set_nbv_pose[2] = 0
                    z_set_nbv_pose = z_set_nbv_pose / z_set_nbv_pose.norm()
                    if (
                        z_cam_pose.dot(z_set_nbv_pose) > 0.8
                        and (cam_pose[:3, 3] - self.set_nbv_pose[:3, 3]).norm() < 0.1
                        and
                        id_len > 2
                    ):
                        recon_signal = True
                if recon_signal:
                    cv2.imwrite(
                        "%s/%06d.png"
                        % (self.exp_path + "/color", self.view_num[0].item()),
                        color_image,
                    )
                    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                    color_image = color_image / 255.0
                    color_image = torch.from_numpy(
                        color_image).float().to(self.device)

                    depth_image = bridge.imgmsg_to_cv2(depth_msg, "16UC1")
                    cv2.imwrite(
                        "%s/%06d.png"
                        % (self.exp_path + "/depth", self.view_num[0].item()),
                        depth_image,
                    )
                    depth_image = depth_image.astype(np.float32) / 1000.0
                    depth_image = torch.from_numpy(depth_image).to(self.device)

                    # save cam_pose as kitti format
                    self.key_poses.append(cam_pose.numpy().flatten())
                    np.savetxt(self.exp_path + "/poses.txt", self.key_poses)

                    self.set_nbv_pose = self.nbv.run(
                        color_image, depth_image, cam_pose.to(self.device)
                    )
                    if self.set_nbv_pose is not None:
                        self.set_nbv_pose = self.set_nbv_pose.cpu()
                        self.planner.camera_pose_callback(cam_pose)
                        self.planner.reset(
                            rospy.Time.now().to_sec(), self.set_nbv_pose)

        if self.set_nbv_pose is not None:
            print("Way to set pose: %d", color_msg.header.stamp.to_sec())
            levelcam2apriltag, pitch_set = self.planner.get_cam_pose_set(
                color_msg.header.stamp.to_sec()
            )
            body2world = self.apriltag2world @ levelcam2apriltag @ self.body2cam
            if check_tensorpose(body2world) is False:
                print("body2world", body2world)
                print("Set Pose is out of safe area!")
            else:
                body_pose_set = PoseStamped()
                body_pose_set.header.frame_id = "map"
                body_pose_set.header.stamp = rospy.Time.now()
                body_pose_set.pose = tensor2rospose(body2world)
                self.pub_body_pose_set.publish(body_pose_set)
                pitch_msg = Float32()
                pitch_msg.data = pitch_set
                self.pub_cam_pitch_set.publish(pitch_msg)
        if self.g_end_signal:
            print("Shutdown SensorROS")
            rospy.signal_shutdown("shutdown")

    def nbv_run(self, args, task):
        self.nbv = NBV(args, task)
        self.planner = Planner(args, task)
        self.set_pose_time = 0

        rospy.init_node("active_recon")
        self.get_state_service = rospy.ServiceProxy(
            "/gazebo/get_model_state", GetModelState
        )
        self.set_state_service = rospy.ServiceProxy(
            "/gazebo/set_model_state", SetModelState
        )

        sub_color = message_filters.Subscriber("/d435/color/image_raw", Image)
        sub_depth = message_filters.Subscriber("/d435/depth/image_raw", Image)
        sub_color_depth = message_filters.ApproximateTimeSynchronizer(
            [sub_color, sub_depth], 10, 0.05
        )
        sub_color_depth.registerCallback(self.cam_callback)

        rospy.spin()

    def nbv_real_run(self, args, task):
        self.nbv = NBV(args, task)
        self.planner = Planner(args, task)

        downward = -0.05
        board = create_board(
            0.09, 0.03, 7, 7, np.array([0.435, 0.435, downward]))
        box = create_box(
            0.0735, 0.0195, 0.013, 0.008, 2, 2, np.array(
                [0.075, 0.075, downward])
        )
        self.board = np.concatenate([board, box], 0)

        self.detector = initDetector()

        self.cam2body = torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.1],
                [0.0, 1.0, 0.0, -0.015],
                [-1.0, 0.0, 0.0, -0.05],
                [0.0000, 0.0000, 0.0000, 1.0000],
            ]
        )
        self.body2cam = self.cam2body.inverse()

        self.apriltag2world = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, downward + 0.003],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        rospy.init_node("active_recon")

        self.pub_body_pose_set = rospy.Publisher(
            "/ar/cam_pose_level_set", PoseStamped, queue_size=10
        )
        self.pub_cam_pitch_set = rospy.Publisher(
            "/ar/cam_pitch_set", Float32, queue_size=10
        )

        sub_color = message_filters.Subscriber(
            "/camera/color/image_raw/compressed", CompressedImage
        )
        sub_depth = message_filters.Subscriber(
            "/camera/aligned_depth_to_color/image_raw", Image
        )

        sub_color_depth = message_filters.TimeSynchronizer(
            [sub_color, sub_depth], 10)
        sub_color_depth.registerCallback(self.real_callback)

        rospy.spin()

    def offline_run(self, args, task):
        self.nbv = NBV(args, task)

        downward = -0.05
        board = create_board(
            0.09, 0.03, 7, 7, np.array([0.435, 0.435, downward]))
        box = create_box(
            0.0735, 0.0195, 0.013, 0.008, 2, 2, np.array(
                [0.075, 0.075, downward])
        )
        self.board = np.concatenate([board, box], 0)

        self.detector = initDetector()

        pose_path = os.path.join(args.exp_path, "poses.txt")
        poses = np.loadtxt(pose_path)

        for i in range(poses.shape[0]):
            color_image = cv2.imread("%s/%06d.png"
                                     % (self.exp_path + "/color", i))
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            color_image = color_image / 255.0
            color_image = torch.from_numpy(
                color_image).float().to(self.device)

            depth_image = cv2.imread("%s/%06d.png"
                                     % (self.exp_path + "/depth", i), -1)
            depth_image = depth_image.astype(np.float32) / 1000.0
            depth_image = torch.from_numpy(depth_image).to(self.device)

            cam_pose = poses[i].reshape(4, 4)
            cam_pose = torch.from_numpy(cam_pose)

            self.set_nbv_pose = self.nbv.run(
                color_image, depth_image, cam_pose.to(self.device)
            )

        if self.g_end_signal:
            print("Shutdown SensorROS")
            rospy.signal_shutdown("shutdown")
