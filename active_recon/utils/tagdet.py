import cv2
from matplotlib import pyplot as plt
import numpy as np
import cv2.aruco as aruco

def creat_side(tag_size, gap_size, gap_size_short, height, width):
    board = []
    group_size = tag_size + gap_size
    for i in range(height):
        for j in range(width):
            tag = []
            bot_left = np.array([j * group_size, 0, i * group_size]) + \
                       np.array([gap_size_short, 0, gap_size_short])
            bot_right = bot_left + np.array([tag_size, 0, 0])
            top_left = bot_left + np.array([0, 0, tag_size])
            top_right = bot_left + np.array([tag_size, 0, tag_size])
            
            tag.append(top_left)
            tag.append(top_right)
            tag.append(bot_right)
            tag.append(bot_left)
            board.append(tag)
            
    board = np.array(board)
    
    return board

def create_box(tag_size, gap_size, gap_size_long, gap_size_short, height, width, center = np.zeros(3)):
    T = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    
    group_size = tag_size + gap_size
    full_size_long = width * group_size - gap_size + 2 * gap_size_long
    full_size_short = width * group_size - gap_size + 2 * gap_size_short
    gap_diff = gap_size_long - gap_size_short
    
    side_1 = creat_side(tag_size, gap_size, gap_size_long, height, width)
    side_1 = side_1.reshape(-1, 3)
    
    side_2 = creat_side(tag_size, gap_size, gap_size_short, height, width)
    side_2 = side_2.reshape(-1, 3)
    side_2 = np.matmul(T, side_2.T).T
    side_2 = side_2 + np.array([full_size_long, 0, gap_diff])
    
    side_3 = creat_side(tag_size, gap_size, gap_size_long, height, width)
    side_3 = side_3.reshape(-1, 3)
    side_3 = np.matmul(T, side_3.T).T
    side_3 = side_3 + np.array([full_size_short, 0, 0])
    side_3 = np.matmul(T, side_3.T).T
    side_3 = side_3 + np.array([full_size_long, 0, 0])
    
    
    side_4 = creat_side(tag_size, gap_size, gap_size_short, height, width)
    side_4 = side_4.reshape(-1, 3)
    side_4 = side_4.reshape(-1, 3)
    side_4 = np.matmul(np.linalg.inv(T), side_4.T).T
    side_4 = side_4 + np.array([0, full_size_short, gap_diff])
    
    board = np.concatenate([side_1, side_2, side_3, side_4], 0)
    board = board.reshape(-1, 4, 3)
    board = board - center
    
    return board

def create_board(tag_size, gap_size, width, height, center = np.zeros(3)):
    board = []
    group_size = tag_size + gap_size
    for i in range(height):
        for j in range(width):
            tag = []
            bot_left = np.array([j * group_size, i * group_size]) + \
                       np.array([gap_size, gap_size])
            bot_right = bot_left + np.array([tag_size, 0])
            top_left = bot_left + np.array([0, tag_size])
            top_right = bot_left + np.array([tag_size, tag_size])
            
            tag.append(top_left)
            tag.append(top_right)
            tag.append(bot_right)
            tag.append(bot_left)
            board.append(tag)
    
    board = np.array(board)
    tag_num = board.shape[0]
    board = np.concatenate([board, np.zeros([tag_num, 4, 1])], -1)
    board = board - center
    
    return board

def initDetector():
    parameters = aruco.DetectorParameters()
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 12
    parameters.adaptiveThreshWinSizeStep = 3
    parameters.adaptiveThreshConstant = 20
    
    # parameters.minMarkerPerimeterRate = 0.05
    # parameters.maxMarkerPerimeterRate = 4.0
    # parameters.polygonalApproxAccuracyRate = 0.05
    # parameters.minCornerDistanceRate = 0.09
    # parameters.minMarkerDistanceRate = 0.03
    # parameters.minDistanceToBorder = 5
    
    # parameters.markerBorderBits = 1
    # parameters.minOtsuStdDev = 5.0
    # parameters.perspectiveRemovePixelPerCell = 4
    # parameters.perspectiveRemoveIgnoredMarginPerCell = 0.13
    
    # parameters.maxErroneousBitsInBorderRate = 0.35
    # parameters.errorCorrectionRate = 0.6
    
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    # parameters.cornerRefinementWinSize = 5
    # parameters.cornerRefinementMaxIterations = 30
    # parameters.cornerRefinementMinAccuracy = 0.1

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
    
    detector = aruco.ArucoDetector(dictionary, parameters)
    
    return detector

def TagDetector(detector, image, board, intrinsic):
    t_tag_coord = []
    t_cam_coord = []
    pose = None

    markerCorners, markerIds, _ = detector.detectMarkers(image)
    
    # print(markerIds)
    
    if markerIds is None:
        id_len = 0
    else:
        id_len = len(markerIds)
        for tag_num in range(len(markerIds)):
            id = markerIds[tag_num][0]
            
            for corner_num in range(4):
                t_tag_coord.append(board[id][corner_num])
                t_cam_coord.append(markerCorners[tag_num][0][corner_num])
        
        _, rvec, tvec, inliers  = cv2.solvePnPRansac(np.array(t_tag_coord), np.array(t_cam_coord), intrinsic, np.array(
                [0.16313493251800537, -0.48739343881607056, -0.0027307835407555103, 0.00011939820979023352, 0.43414241075515747]))
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3)
        R_eigen = np.array(R, dtype=np.float32)
        t_eigen = np.array(t, dtype=np.float32)
        pose = np.eye(4)
        pose[:3, :3] = R_eigen.transpose()
        pose[:3, 3] = -R_eigen.transpose() @ t_eigen
        # # vis
        # aruco.drawDetectedMarkers(image, markerCorners)
        # cv2.imshow("image", image)
        # cv2.waitKey()
        
    return id_len,pose

if __name__ == "__main__":
    tag_image = cv2.imread("/home/nrosliu/Projects/Active_Recon/ActiveRecon/exp/realworld_6_box_noposeopt/color/000002.png")
    downward = -0.05
    board = create_board(
        0.09, 0.03, 7, 7, np.array([0.435, 0.435, downward]))
    box = create_box(
        0.0735, 0.0195, 0.013, 0.008, 2, 2, np.array(
            [0.075, 0.075, downward])
    )
    board = np.concatenate([board, box], 0)
    
    # board = board.reshape(-1,3)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.scatter(board[:,0], board[:,1],board[:,2])
    # # plt.plot()
    # size = 0.5
    # ax.set_xlim(-size, size)
    # ax.set_ylim(-size, size)
    # ax.set_zlim(-size, size)
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    # plt.show()
    
    detector = initDetector()
    
    intrinsic = np.array([[600.6603393554688, 0.0, 330.6833190917969], 
                          [0.0, 600.7969360351562, 234.0941619873047], 
                          [0.0, 0.0, 1.0]])
    
    num_inlier, pose = TagDetector(detector, tag_image, board, intrinsic)
    
    print(pose)
    print(np.linalg.inv(pose))
    