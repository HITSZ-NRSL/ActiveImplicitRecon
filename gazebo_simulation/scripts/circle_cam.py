from nav_msgs.msg import Path

from geometry_msgs.msg import PoseStamped
import rospy

import math

from tf.transformations import quaternion_from_euler  # 欧拉转四元数

from gazebo_msgs.srv import SetModelState, SetModelStateRequest


def pub_o_path():
    set_state_service = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
    # 定义发布者，应该注意的是发布的消息类型始终为Path，Path之下的消息类型geometry_msgs/PoseStamped只是用于赋值，而不是使用PoseStamped发布

    pub_path = rospy.Publisher("path_pubulisher", Path, queue_size=10)

    msg = Path()

    # 指定frame_id和stamp，frame_id是在rviz显示的时候会用到，需要在rviz中fixed frame位置输入你定义的frame_id，这里使用rviz默认的map

    # stamp是时间辍，看了很多例子一般是使用rospy.Time.now()，不知道还有没有其他的设定，挖个坑。

    msg.header.frame_id = "map"

    msg.header.stamp = rospy.Time.now()

    rate = rospy.Rate(10)

    # 由于是定义了一个做圆周的物体，因此还需要给它一些初始值

    o_z = 0.2

    r = 1

    t = 0
    
    v = 0.01

    # 开始循环发布消息

    while not rospy.is_shutdown():

        # 定义求出圆的运动方程，并保存经过的点

        # 定义一个变量对PoseStamped进行赋值

        # 再次强调，应该注意的是发布的消息类型始终为Path，Path之下的消息类型geometry_msgs/PoseStamped只是用于赋值，而不是使用PoseStamped发布

        pose = PoseStamped()

        pose.pose.position.x = -r * math.cos(t)

        pose.pose.position.y = -r * math.sin(t)

        pose.pose.position.z = o_z

        # 由于th是用欧拉角表示，而PoseStamped中是用四元数表示角度用的，因此需要将th转换为四元数表示

        quaternion = quaternion_from_euler(0, 0, t)

        pose.pose.orientation.x = quaternion[0]

        pose.pose.orientation.y = quaternion[1]

        pose.pose.orientation.z = quaternion[2]

        pose.pose.orientation.w = quaternion[3]

        # 之前提到过PoseStamped消息类型是以列表的形式保存，因此需要将坐标和角度信息保存保存至msg中：

        # nav_msgs/Path数据类型

        # Header header

        # geometry_msgs/PoseStamped[] poses

        msg.poses.append(pose)

        # 发布消息

        pub_path.publish(msg)

        rospy.loginfo("x = {}, y = {}, th = {}".format(-r * math.cos(t), -r * math.sin(t), t))

        cam_state = SetModelStateRequest()
        cam_state.model_state.model_name = "d435_model"
        cam_state.model_state.pose = pose.pose
        cam_state.model_state.reference_frame = "world"
        set_state_service(cam_state)

        t += v

        rate.sleep()


if __name__ == "__main__":
    rospy.init_node("show_path", anonymous=True)
    try:
        pub_o_path()
    except rospy.ROSInterruptException:
        pass
