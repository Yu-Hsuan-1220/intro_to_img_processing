import cv2
import numpy as np
from shapely.geometry import Polygon, Point
import os



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
        
def solve_homography(src_point, des_point):
    A = []
    B = []

    for (x, y), (x_prime, y_prime) in zip(src_point, des_point):
        A.append([x, y, 1, 0, 0, 0, -x*x_prime, -y*x_prime])
        A.append([0, 0, 0, x, y, 1, -x*y_prime, -y*y_prime])
        B.append(x_prime)
        B.append(y_prime)

    A = np.array(A)
    b = np.array(B)
    h, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return h

def apply_homography(H, x, y):
    h11, h12, h13 = H[0, :]
    h21, h22, h23 = H[1, :]
    h31, h32 = H[2, 0], H[2, 1]

    deno = h31*x + h32*y + 1
    x_prime = (h11 * x + h12 * y + h13) / deno
    y_prime = (h21 * x + h22 * y + h23) / deno

    return x_prime, y_prime

image_path = './HW1/image.jpg'
board_path = './HW1/board.jpg'
result_dir = './HW1/result'

img = cv2.imread(image_path)
board = cv2.imread(board_path)

h, w, c = img.shape
des_point = np.array([[0, 0], [h-1, 0], [h-1, w-1], [0, w-1]])
src_point = np.array([[241, 253], [375, 253], [387, 413], [215, 413]])
H = solve_homography(src_point, des_point)

H = np.array([[H[0], H[1], H[2]], [H[3], H[4], H[5]], [H[6], H[7], 1]])


#print(H_inv)


trapezoid = Polygon([(241, 252), (375, 252), (387, 413), (215, 413)])
x_min, y_min, x_max, y_max = trapezoid.bounds
inside_points = [(x, y) for x in range(int(x_min), int(x_max) + 1)
                          for y in range(int(y_min), int(y_max) + 1)
                          if trapezoid.contains(Point(x, y))]

for (i, j) in inside_points:
    x, y = apply_homography(H, i, j)
    if in_bound(x, y, h, w): 
        board[i, j, :] = bicubic_interpolation(img, x, y)



# for i in range(h):
#     for j in range(w):
#         x, y = apply_homography(H, i, j)
#         board[int(x), int(y), :] = img[i, j, :]


cv2.imwrite(os.path.join(result_dir, 'Bicubic_homography.jpg'), board)