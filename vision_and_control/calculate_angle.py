import math
import numpy as np

# 舵机座中心 + 支腿底部 (x, y, z) = (0, 0, 0)
# 目标点坐标与六爪底部中心相同
# 设与支腿面平行的方向（支腿弯曲的方向）为x正方向
# x正方向水平角度和竖直角度均为0
#	水平角度和竖直角度范围为[-180, 180)
# 所有长度单位为cm

# 目标值：
# angle_hor - 舵轴旋转角度
# angle_major_arm - 大臂主轴旋转角度
# angle_bent_arm - 曲柄轴旋转角度

# 预设值：
# leg_height - 支腿底部到大臂主轴/曲柄轴的高度
# major_arm_len - 大臂长度（以两端轴心为准测量，下同）
# minor_arm_len_m - 小臂长度1（以和大臂连接的轴为准）
# minor_arm_len_b - 小臂长度2（以和曲柄连接的轴为准）
# minor_arm_len_d - 小臂长度3（三角形短边长度）
# bent_arm_len - 曲柄长度
# bent_stick_len - 曲柄连杆长度
# plane_dist_h - 【小臂与水平座连接处的轴心】与【六爪中心底部】的水平距离
# plane_dist_v - 【小臂与水平座连接处的轴心】与【六爪中心底部】的竖直距离
# steer_offset_x - 舵轴到大臂轴中线的距离（与大臂平行的方向）
# steer_offset_y - 舵轴到大臂轴中线的距离（与大臂垂直的方向）

leg_height = 15.5
major_arm_len = 32
minor_arm_len_m = 24
minor_arm_len_b = 27
minor_arm_len_d = 9
bent_arm_len = 8
bent_stick_len = 28
plane_dist_h = 4
plane_dist_v = 3
steer_offset_x = 2
steer_offset_y = 1

def calc_angle(x_pos, y_pos, z_pos):
	# 计算舵轴旋转角度
	angle_h = math.atan2(y_pos, x_pos)

	# 计算大臂主轴及小臂旋转角度
	angle_ma, angle_tmp = helper_1(angle_h, x_pos, y_pos, z_pos)

	angle_h, angle_ma, angle_tmp = numerical_sol(angle_h, angle_ma, angle_tmp, x_pos, y_pos, z_pos)

	# 根据小臂旋转角度计算曲柄轴旋转角度
	angle_ba = helper_2(angle_tmp)

	angle_h_deg = math.degrees(angle_h)
	angle_ma_deg = math.degrees(angle_ma)
	angle_ba_deg = math.degrees(angle_ba)
	return angle_h_deg, angle_ma_deg, angle_ba_deg

# 计算大臂主轴旋转角及小臂旋转角度
def helper_1(angle_1, px, py, pz):
	a2 = major_arm_len
	a3 = minor_arm_len_m

	# 计算小臂旋转角度
	c3 = ((px - math.cos(angle_1)*plane_dist_h)**2 + (py - math.sin(angle_1)*plane_dist_h)**2
		+ (pz + plane_dist_v - leg_height)**2 - a2**2 - a3**2) / (2*a2*a3)
	s3 = -math.sqrt(1 - c3**2)
	angle_3 = math.atan2(s3, c3)

	# 计算大臂主轴旋转角度
	s2_t = ((pz + plane_dist_v - leg_height)*(a2 + a3*c3)
		- a3*s3*math.sqrt((px - math.cos(angle_1)*plane_dist_h)**2 + (py - math.sin(angle_1)*plane_dist_h)**2))
	c2_t = ((-1)*(pz + plane_dist_v - leg_height)*a3*s3
		+ (a2 + a3*c3)*math.sqrt((px - math.cos(angle_1)*plane_dist_h)**2 + (py - math.sin(angle_1)*plane_dist_h)**2))
	angle_2 = math.atan2(s2_t, c2_t)

	return angle_2, angle_3

def helper_2(angle_arm):
	# 计算小臂三个轴所构成的三角形的钝角（顶点为大小臂连接轴）
	angle_minor = math.acos((minor_arm_len_d**2 + minor_arm_len_m**2
		- minor_arm_len_b**2)/(2*minor_arm_len_d*minor_arm_len_m))

	# 将曲柄轴（大臂主轴）与曲柄和小臂的连接轴相连，构成两个三角形
	# 解含大臂的三角形
	angle_4 = math.pi - angle_minor + angle_arm
	a4 = math.sqrt(major_arm_len**2 + minor_arm_len_d**2
		- 2*major_arm_len*minor_arm_len_d*math.cos(angle_4))
	angle_4_1 = math.acos((a4**2 + major_arm_len**2
		- minor_arm_len_d**2)/(2*a4*major_arm_len))

	# 解含曲柄的三角形
	angle_5 = math.acos((a4**2 + bent_arm_len**2
		- bent_stick_len**2)/(2*a4*bent_arm_len))

	angle_ret = angle_4_1 + angle_5
	return angle_ret

# 在解析解的基础上用迭代法算出数值解
def numerical_sol(angle_1, angle_2, angle_3, x_t, y_t, z_t):
	# Learning rate
	alpha = 1e-4

	# Initialization
	angles = np.zeros(3)
	angles[0] = angle_1
	angles[1] = angle_2
	angles[2] = angle_3
	target_pos = np.zeros(3)
	target_pos[0] = x_t
	target_pos[1] = y_t
	target_pos[2] = z_t
	calc_pos = forward_calc(angles[0], angles[1], angles[2])
	diff = target_pos - calc_pos

	# 迭代
	while (np.linalg.norm(diff) >= 0.5):
		J_T = Jacobian_T(angles)
		d_angle = alpha*np.dot(J_T, diff)
		angles += d_angle

		# 修正角度范围到[-pi, pi)
		angles[0] = angles[0] % (2*math.pi)
		angles[1] = angles[1] % (2*math.pi)
		angles[2] = angles[2] % (2*math.pi)

		if (angles[0] >= math.pi):
			angles[0] -= 2*math.pi
		if (angles[1] >= math.pi):
			angles[1] -= 2*math.pi
		if (angles[2] >= math.pi):
			angles[2] -= 2*math.pi

		calc_pos = forward_calc(angles[0], angles[1], angles[2])
		diff = target_pos - calc_pos

	return angles[0], angles[1], angles[2]

# 正向运动学计算
def forward_calc(angle_1, angle_2, angle_3):
	hor_dist = (major_arm_len*math.cos(angle_2)
		+ minor_arm_len_m*math.cos(angle_2+angle_3)
		+ plane_dist_h + steer_offset_x)
	offset_angle = math.atan2(steer_offset_y, hor_dist)
	pos = np.zeros(3)
	pos[0] = math.cos(angle_1 + offset_angle)*math.sqrt(hor_dist**2 + steer_offset_y**2)
	pos[1] = math.sin(angle_1 + offset_angle)*math.sqrt(hor_dist**2 + steer_offset_y**2)
	pos[2] = major_arm_len*math.sin(angle_2) + minor_arm_len_m*math.sin(angle_2+angle_3) - plane_dist_v + leg_height
	return pos

# Transposed Jacobian matrix
def Jacobian_T(angles):
	angle_1 = angles[0]
	angle_2 = angles[1]
	angle_3 = angles[2]

	J = np.zeros((3, 3))
	# dx/d(angle_1)
	J[0][0] = (-1)*math.sin(angle_1)*math.sqrt((major_arm_len*math.cos(angle_2)
		+ minor_arm_len_m*math.cos(angle_2+angle_3) + plane_dist_h + steer_offset_x)**2 + steer_offset_y**2)
	# dx/d(angle_2)
	J[0][1] = (math.cos(angle_1)*((-1)*major_arm_len*math.sin(angle_2) + (-1)*minor_arm_len_m*math.sin(angle_2+angle_3))
		/math.sqrt((major_arm_len*math.cos(angle_2) + minor_arm_len_m*math.cos(angle_2+angle_3)
			+ plane_dist_h + steer_offset_x)**2 + steer_offset_y**2))
	# dx/d(angle_3)
	J[0][2] = (math.cos(angle_1)*((-1)*minor_arm_len_m*math.sin(angle_2+angle_3))
		/math.sqrt((major_arm_len*math.cos(angle_2) + minor_arm_len_m*math.cos(angle_2+angle_3)
			+ plane_dist_h + steer_offset_x)**2 + steer_offset_y**2))
	# dy/d(angle_1)
	J[1][0] = math.cos(angle_1)*math.sqrt((major_arm_len*math.cos(angle_2)
		+ minor_arm_len_m*math.cos(angle_2+angle_3) + plane_dist_h + steer_offset_x)**2 + steer_offset_y**2)
	# dy/d(angle_2)
	J[1][1] = (math.sin(angle_1)*((-1)*major_arm_len*math.sin(angle_2) + (-1)*minor_arm_len_m*math.sin(angle_2+angle_3))
		/math.sqrt((major_arm_len*math.cos(angle_2) + minor_arm_len_m*math.cos(angle_2+angle_3)
			+ plane_dist_h + steer_offset_x)**2 + steer_offset_y**2))
	# dy/d(angle_3)
	J[1][2] = (math.sin(angle_1)*((-1)*minor_arm_len_m*math.sin(angle_2+angle_3))
		/math.sqrt((major_arm_len*math.cos(angle_2) + minor_arm_len_m*math.cos(angle_2+angle_3)
			+ plane_dist_h + steer_offset_x)**2 + steer_offset_y**2))
	# dz/d(angle_1)
	J[2][0] = 0
	# dz/d(angle_2)
	J[2][1] = major_arm_len*math.cos(angle_2) + minor_arm_len_m*math.cos(angle_2+angle_3)
	# dz/d(angle_3)
	J[2][2] = minor_arm_len_m*math.cos(angle_2+angle_3)
	return J.T

# 主函数
x, y, z = map(float, input("What's the (x, y, z) position of the object? Like 10 20 30. ").split(' '))
angle_hor, angle_major_arm, angle_bent_arm = calc_angle(x, y, z)
print("舵轴旋转角度: %f°" %(angle_hor))
print("大臂主轴旋转角度: %f°" %(angle_major_arm))
print("曲柄轴旋转角度: %f°" %(angle_bent_arm))