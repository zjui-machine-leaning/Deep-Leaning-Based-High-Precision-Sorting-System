import cv2
import numpy as np
from xml.dom.minidom import Document
from xml.dom.minidom import parse
from math import *
import os
import random
import sys

class OBJ:
    def __init__(self, name, pts, height, width):
        self.name = name
        self.pts = pts
        self.height = height
        self.width = width

def writeXML(filename, objs, imgsize):
    doc = Document()
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)
    
    # imagesize
    imagesize = doc.createElement('imagesize')
    # nrows
    nrows = doc.createElement('nrows')
    nrows_text = doc.createTextNode(str(imgsize[0]))
    nrows.appendChild(nrows_text)
    imagesize.appendChild(nrows)
    # ncols
    ncols = doc.createElement('ncols')
    ncols_text = doc.createTextNode(str(imgsize[1]))
    ncols.appendChild(ncols_text)
    imagesize.appendChild(ncols)
    annotation.appendChild(imagesize)
    
    i = 0  # Counter
    for item in objs:
        obj = doc.createElement('object')
        annotation.appendChild(obj)

        # name
        name = doc.createElement('name')
        name_text = doc.createTextNode(item.name)
        name.appendChild(name_text)
        obj.appendChild(name)
        # id
        obj_id = doc.createElement('id')
        id_text = doc.createTextNode(str(i))
        obj_id.appendChild(id_text)
        obj.appendChild(obj_id)

        polygon = doc.createElement('polygon')
        for p in item.pts:
            pt = doc.createElement('pt')
            x = doc.createElement('x')
            x_text = doc.createTextNode(str(p[0]))
            x.appendChild(x_text)
            pt.appendChild(x)
            y = doc.createElement('y')
            y_text = doc.createTextNode(str(p[1]))
            y.appendChild(y_text)
            pt.appendChild(y)
            polygon.appendChild(pt)

        obj.appendChild(polygon)
        i += 1

    with open(filename, 'w') as f:
        doc.writexml(f,indent='',addindent='\t',newl='\n',encoding='UTF-8')

def readObjXML(filename):
    doc = parse(filename)
    root = doc.documentElement
    # Get object name
    name = root.getElementsByTagName("name")[0].childNodes[0].data
    # Get object height and width
    height = root.getElementsByTagName("nrows")[0].childNodes[0].data
    width = root.getElementsByTagName("ncols")[0].childNodes[0].data
    # Get point list
    pts = []
    all_pts = root.getElementsByTagName("pt")
    for pt in all_pts:
        x = pt.getElementsByTagName("x")[0].childNodes[0].data
        y = pt.getElementsByTagName("y")[0].childNodes[0].data
        pts.append([int(x),int(y)])
    
    # Return the OBJ item
    return OBJ(name,pts,height,width)

def produceImg(bg_name, prop_list, img_id):
    # Create object list
    obj_lst = []
    # Read background image
    dst = cv2.imread(bg_name)

    for prop in prop_list:
        # Read object image
        src = cv2.imread(prop[0])
        # Read object XML
        obj_tmp = readObjXML(prop[0][:-3]+"xml")
        
        # Resize the object
        if prop[1] != 1:
            src = cv2.resize(src, None, fx=prop[1], fy=prop[1])
            # Adjust the annotation of the object
            for i in range(len(obj_tmp.pts)):
                obj_tmp.pts[i][0] = int(prop[1]*obj_tmp.pts[i][0])
                obj_tmp.pts[i][1] = int(prop[1]*obj_tmp.pts[i][1])
              
        # Rotate hte object
        if prop[2] != 0:
            orig_src_width = src.shape[1]
            orig_src_height = src.shape[0]
            src = rotate_bound(src, prop[2])
            # Adjust the annotation of the object
            for i in range(len(obj_tmp.pts)):
                # Adjust the rotate center to (0,0)
                obj_tmp.pts[i][0] -= orig_src_width/2
                obj_tmp.pts[i][1] -= orig_src_height/2
                # Rotate the points
                orig_x = obj_tmp.pts[i][0]
                orig_y = obj_tmp.pts[i][1]
                obj_tmp.pts[i][0] = orig_x*cos(radians(prop[2])) + orig_y*sin(radians(prop[2]))
                obj_tmp.pts[i][1] = -orig_x*sin(radians(prop[2])) + orig_y*cos(radians(prop[2]))
                # Center to the resulting image
                obj_tmp.pts[i][0] = int(obj_tmp.pts[i][0] + src.shape[1]/2)
                obj_tmp.pts[i][1] = int(obj_tmp.pts[i][1] + src.shape[0]/2)
                
        # Create a rough mask around the object.
        srcgray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
        src_mask = np.zeros(srcgray.shape, src.dtype)

        poly = np.array(obj_tmp.pts, np.int32)
        cv2.fillPoly(src_mask, [poly], 255)

        # The center of the object in the background (row, col)
        src_height, src_width = src.shape[0], src.shape[1]
        row_min = src_height/2
        row_max = dst.shape[0] - src_height/2
        col_min = src_width/2
        col_max = dst.shape[1] - src_width/2
        center_x = np.random.randint(row_min, row_max+1)
        center_y = np.random.randint(col_min, col_max+1)
        center = (center_x, center_y)
        # Get the position of the object in background
        bg_rowbound = center[0] - int(row_min)
        bg_colbound = center[1] - int(col_min)
        for i in range(len(obj_tmp.pts)):
            obj_tmp.pts[i][0] += bg_colbound  # Col offset
            obj_tmp.pts[i][1] += bg_rowbound  # Row offset
        

        mask_inv = cv2.bitwise_not(src_mask)
        bg_roi = dst[bg_rowbound:(bg_rowbound+src_height), bg_colbound:(bg_colbound+src_width)]
        
        # Append the OBJ item to object list
        obj_lst.append(obj_tmp)

        
# Replace the new region
        bg = cv2.bitwise_and(bg_roi,bg_roi,mask = mask_inv)
        obj_img = cv2.bitwise_and(src,src,mask=src_mask)
        new_region = cv2.add(bg,obj_img)
        dst[bg_rowbound:(bg_rowbound+src_height), bg_colbound:(bg_colbound+src_width)] = new_region
    # Save the resulting image
    cv2.imwrite("output/data_image/"+str(img_id)+".jpg", dst)
    # Return the object list
    return obj_lst

def file_name(file_dir):
    files = os.listdir(file_dir)
    ret = []
    for file in files:
        if file.endswith('jpg'):
            ret.append(file_dir+"/"+file)
    return ret

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def main(argv):
    err_msg = '''Please enter command in the following form:
    python create_data.py [num_output] [-n]/[-c]
    where -n means to create a new dataset, and -c means to add to the previous dataset.'''
    
# Check the arguments
    if len(argv) != 2:
        print(err_msg)
        return
    elif argv[1] != '-n' and argv[1] != '-c':
        print(err_msg)
        return

    # Get the base data index
    output_list = file_name("output")
    if argv[1] == '-n':
        base = 0
        for file in output_list:
            os.remove(file)
            os.remove(file[:-3]+"xml")
    else:
        if len(output_list) == 0:
            base = 0
        else:
            output_list.sort(key = lambda x:int(x[7:-4]))
            base = int(output_list[-1][7:-4])
        
    battery_list = file_name("battery")
    rpcan_list = file_name("ring-pull can")
    bg_list = file_name("background")

    n_output = int(argv[0])
    for i in range(1,n_output+1):
        background = random.choice(bg_list)
        n_battery = random.randint(3,5)
        n_rpcan = random.randint(3,5)
        batteries = [random.choice(battery_list) for i in range(n_battery)]
        rpcans = [random.choice(rpcan_list) for i in range(n_rpcan)]
        prop_list = []
        for battery in batteries:
            scale = 0.1*random.randint(7,12)
            angle = 10*random.randint(1,35)
            prop_list.append((battery, scale, angle))
        for rpcan in rpcans:
            scale = 0.1*random.randint(7,12)
            angle = 10*random.randint(1,35)
            prop_list.append((rpcan, scale, angle))
        bg_img = cv2.imread(background)
        obj_list = produceImg(background, prop_list, base+i)
        print("Data ID:",base+i)
        writeXML("output/data_mask/"+str(base+i)+".xml", obj_list,
                 (bg_img.shape[0], bg_img.shape[1]))

if __name__ == '__main__':
    main(sys.argv[1:])
