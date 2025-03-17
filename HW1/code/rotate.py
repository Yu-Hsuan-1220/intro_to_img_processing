import cv2 
import numpy as np
import os

path = './HW1/board.jpg'
result_dir = './HW1/result'

def in_bound(i, j, h, w):
    return i>=0 and i<h and j>=0 and j<w

def function_val(p0, p1, p2, p3, x):
    return np.clip((-0.5*p0 + 1.5*p1 - 1.5*p2 + 0.5*p3)*(x**3) + (p0 - 2.5*p1 + 2*p2 - 0.5*p3)*(x**2) + (-0.5*p0 + 0.5*p2)*(x)+ p1, 0, 255)

def bicubic_interpolation(img, src_x, src_y):
    h, w, c = img.shape
    x = int(src_x)
    y = int(src_y)
    result = np.zeros(3)
    for channel in range(c):
        ver_val = []
        for i in range(-1, 3):
            hor_val = []
            px = np.clip(x+i, 0, h-1)     
            for j in range(-1, 3):
                py = np.clip(y+j, 0, w-1)
                hor_val.append(img[px, py, channel])
            ver_val.append(function_val(hor_val[0], hor_val[1], hor_val[2], hor_val[3], src_y-y))
        result[channel] = np.clip(function_val(ver_val[0], ver_val[1], ver_val[2], ver_val[3], src_x-x), 0, 255)
    return result.astype(np.uint8)

def NN_interpolation(img, src_x, src_y):
    x = int(src_x)
    y = int(src_y)
    return img[x, y, :]

def Bilinear_interpolation(img, src_x, src_y):
    x = int(src_x)
    y = int(src_y)
    h, w, c = img.shape

    dx = src_x - x
    dy = src_y - y

    if x == h-1 or y == w-1:
        return img[x, y]

    tmp1 = img[x, y] * (1-dy) + img[x, y+1] * (dy)
    tmp2 = img[x+1, y] * (1-dy) + img[x+1, y+1] * (dy)

    return (tmp1 * (1-dx) + tmp2 * dx) 
        


img = cv2.imread(path)

h, w, c = img.shape

interpolation = True

emp_img = np.zeros((h, w, c), dtype=np.uint8)

theta = np.pi/6

rotate_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

center_x, center_y = h // 2, w // 2

for i in range(h):
    for j in range(w):
        vec_i, vec_j = i-center_x, j-center_y
        src_x, src_y = rotate_mat @ np.array([vec_i, vec_j])
        src_x = center_x + src_x
        src_y = center_y + src_y
        if in_bound(src_x, src_y, h, w):
            emp_img[i, j, :] = bicubic_interpolation(img, src_x, src_y) if interpolation == True else img[int(src_x), int(src_y), :]


cv2.imwrite(os.path.join(result_dir, 'rotate_bicubic.jpg'), emp_img)
cv2.waitKey(0)


