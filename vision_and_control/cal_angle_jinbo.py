import math

def func(x,y,z):   
    b = 10
    c = 38.5
    d = 34.5
    e = 11
    f = 45.5
    steer_offset_x = 5
    steer_offset_y = 10

    # 3d version


    tan_c = y/x

    angle_c = math.atan(tan_c)

    # 2d version

    x_2d = (x**2 + y**2)**0.5
    y_2d = z
    #print("x_2d, y_2d: ", x_2d, y_2d)

    cos_delta = (-x_2d**2-y_2d**2 + c**2 + f**2) / (2*c*f)
    
    angle_delta = math.acos(cos_delta)
    
    #print("angle_delta: ", angle_delta/math.pi*180)

    sin_delta = (1 - cos_delta**2)**0.5

    # sin_180_minus_delta_minus_theta = (sin_theta/f * c)
    
    # angle_180_minus_delta_minus_theta = math.asin(sin_180_minus_delta_minus_theta) = math.asin((sin_theta/f * c))

    ## cos_180_minus_delta_minus_theta = (1- sin_180_minus_delta_minus_theta**2)**0.5 = (1- (sin_theta/f * c)**2)**0.5

    # cos_theta = (1 - sin_theta**2)**0.5

    # sin_delta = sin_theta* (1- (sin_theta/f * c)**2)**0.5 + cos_theta* (sin_theta/f * c) = sin_theta* (1- (sin_theta/f * c)**2)**0.5 + (1 - sin_theta**2)**0.5* (sin_theta/f * c)

    # to be solved

    diff = math.pi
    min_angle_theta = 0
    for i in range(10000):
        angle_theta = i*math.pi/10000
        sin_theta = math.sin(angle_theta)
        # print(sin_theta, abs(sin_delta - ( sin_theta* (1- (sin_theta/f * c)**2)**0.5 + (1 - sin_theta**2)**0.5* (sin_theta/f * c))))
        # print(math.asin((sin_theta/f * c)), angle_theta, angle_delta, abs(math.asin((sin_theta/f * c)) + angle_theta + angle_delta - math.pi), diff)
        if abs(math.asin((sin_theta/f * c)) + angle_theta + angle_delta - math.pi) < diff:
            min_angle_theta = angle_theta
            diff = abs(math.asin((sin_theta/f * c)) + angle_theta + angle_delta - math.pi)

    angle_theta = min_angle_theta
    
    #print(angle_theta/math.pi*180)

    h = (b**2 + c**2 + 2*b*c* math.cos(angle_delta))**0.5
    
    #print("h", h)

    cos_r_one = (h**2 + e**2 - d**2) / (2*h*e)

    sin_r_one = (1- cos_r_one**2)**0.5

    cos_r_two = (h**2 + c**2 - b**2) / (2*h*c)

    sin_r_two = (1- cos_r_two**2)**0.5
    
    angle_r_one = math.acos(cos_r_one)
    
    angle_r_two = math.acos(cos_r_two)

    cos_r = cos_r_one*cos_r_two - sin_r_two * sin_r_one

    angle_r = math.acos(cos_r)
    
    #print("angle_r_one: ",angle_r_one/math.pi*180)
    
    #print("angle_r_two: ", angle_r_two/math.pi*180)
    
    #print("angle_r: ", angle_r*180/math.pi)
    
    tan_a_minus_theta = y_2d/x_2d
    
    #sin_a_minus_theta = (1/((1/tan_a_minus_theta)**2))**0.5

    #cos_a_minus_theta = (1/(1 + tan_a_minus_theta**2))**0.5

    angle_a_minus_theta = math.atan(tan_a_minus_theta)
    
    #print("angle_a_minus_theta: ", angle_a_minus_theta*180/math.pi)

    angle_a = angle_a_minus_theta + angle_theta

    angle_b = angle_a+angle_r

    return (angle_a/math.pi*180, angle_b/math.pi*180, angle_c/math.pi*180)
