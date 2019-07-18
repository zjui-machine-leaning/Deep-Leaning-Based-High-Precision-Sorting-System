# filename: camera_configs.py
import cv2
import numpy as np

left_camera_matrix = np.array([[873.91955, 0., 664.81134],
                               [0., 882.83139, 377.47712],
                               [0., 0., 1.]])
left_distortion = np.array([[0.099821, -0.20512, 0.01175, 0.0185, 0.00000]])



right_camera_matrix = np.array([[876.21501, 0., 656.29572],
                                [0., 879.83518, 379.13359],
                                [0., 0., 1.]])
right_distortion = np.array([[0.05233, -0.06723, 0.00991, -0.01297, 0.00000]])

om = np.array([-0.00577, 0.02201, -0.00138]) # 旋转关系向量
R = cv2.Rodrigues(om)[0]  # 使用Rodrigues变换将om变换为R
T = np.array([-166.75181, -1.91306, -10.16051]) # 平移关系向量

size = (1280, 720) # 图像尺寸

# 进行立体更正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)
# 计算更正map
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)
