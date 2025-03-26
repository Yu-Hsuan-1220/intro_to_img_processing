import cv2
import numpy as np


def equalize_map(img):
    h, w = img.shape
    cnt = [0] * 256
    for i in range (h):
        for j in range(w):
            cnt[img[i,j]] += 1
    
    result_map = []
    cur = 0
    for i in range(256):
        cur += cnt[i]
        result_map.append(round((cur/(h*w))*255))
    return result_map

def get_reverse_map(img):
    equalized_map = equalize_map(img)
    result_map = []
    cur_val = 0
    idx = 0
    for i in range(256): ## Find the value correspond to each Sk(0~255)
        if equalized_map[i] > cur_val:
            while idx < equalized_map[i]:
                result_map.append(cur_val)
                idx += 1
            cur_val = i
    while idx < 256:
        result_map.append(cur_val)
        idx+=1
    return result_map


Q1_img = cv2.imread('Q1.jpeg', cv2.IMREAD_GRAYSCALE)
Q2_ref = cv2.imread('Q2_reference.jpg', cv2.IMREAD_GRAYSCALE)
Q2_src = cv2.imread('Q2_source.jpg', cv2.IMREAD_GRAYSCALE)

## Part 1
result_map = equalize_map(Q1_img)
Q1_result = np.zeros_like(Q1_img)
h, w, = Q1_img.shape
for i in range(h):
    for j in range(w):
        Q1_result[i, j] = result_map[Q1_img[i,j]]
cv2.imwrite('Q1_result.jpg', Q1_result)

## Part 2
result_map = equalize_map(Q2_src)
reversed_map = get_reverse_map(Q2_ref)

h, w = Q2_src.shape
Q2_result = np.zeros_like(Q2_src)
for i in range(h):
    for j in range(w):
        Q2_result[i, j] = reversed_map[result_map[Q2_src[i, j]]]

cv2.imwrite('Q2_result.jpg', Q2_result)
