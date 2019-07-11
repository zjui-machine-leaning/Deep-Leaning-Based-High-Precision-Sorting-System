import math

def func(x,y,z):   
    b = 10.5
    c = 38.5
    d = 35
    e = 10.5
    f = 46
    steer_offset_x = 5
    steer_offset_y = 10

    # 3d version


    tan_c = y/x

    angle_c = math.atan(tan_c)

    # 2d version

    x_2d = (x**2 + y**2)**0.5
    y_2d = z

    cos_delta = (x**2+y**2 - c**2 - f**2) / (2*c*f)

    sin_delta = (1 - cos_delta**2)**0.5

    # sin_180_minus_delta_minus_theta = (sin_theta/f * c)

    # cos_180_minus_delta_minus_theta = (1- sin_180_minus_delta_minus_theta**2)**0.5 = (1- (sin_theta/f * c)**2)**0.5

    # cos_theta = (1 - sin_theta**2)**0.5

    # sin_delta = sin_theta* (1- (sin_theta/f * c)**2)**0.5 + cos_theta* (sin_theta/f * c) = sin_theta* (1- (sin_theta/f * c)**2)**0.5 + (1 - sin_theta**2)**0.5* (sin_theta/f * c)

    # to be solved

    diff = 1
    min_sin_theta = 0
    for i in range(1000):
        sin_theta = i/1000
        # print(sin_theta, abs(sin_delta - ( sin_theta* (1- (sin_theta/f * c)**2)**0.5 + (1 - sin_theta**2)**0.5* (sin_theta/f * c))))
        if abs(sin_delta - ( sin_theta* (1- (sin_theta/f * c)**2)**0.5 + (1 - sin_theta**2)**0.5* (sin_theta/f * c))) < diff:
            min_sin_theta = sin_theta
            diff = sin_delta - ( sin_theta* (1- (sin_theta/f * c)**2)**0.5 + (1 - sin_theta**2)**0.5* (sin_theta/f * c))

    sin_theta = min_sin_theta
    
    #print(sin_theta)

    angle_theta = math.asin(min_sin_theta)

    h = (b**2 + c**2 + 2*b*c* cos_delta)**0.5

    cos_r_one = (h**2 + e**2 - d**2) / (2*h*e)

    sin_r_one = (1- cos_r_one**2)**0.5

    cos_r_two = (h**2 + c**2 - b**2) / (2*h*c)

    sin_r_two = (1- cos_r_two**2)**0.5

    cos_r = cos_r_one*cos_r_two - sin_r_two * sin_r_one

    angle_r = math.acos(cos_r)
    
    tan_a_minus_theta = y/x

    #sin_a_minus_theta = (1/((1/tan_a_minus_theta)**2))**0.5

    #cos_a_minus_theta = (1/(1 + tan_a_minus_theta**2))**0.5

    angle_a_minus_theta = math.atan(tan_a_minus_theta)

    angle_a = angle_a_minus_theta + angle_theta

    angle_b = angle_a+angle_r

    return (angle_a/math.pi*180, angle_b/math.pi*180, angle_c/math.pi*180)