import cv2
import numpy as np
import os
def convolve2d(image, kernel):

    img_h, img_w = image.shape
    k_h, k_w = kernel.shape
    #Caculate padding size
    pad_h = k_h // 2
    pad_w = k_w // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    output = np.zeros_like(image)
    for i in range(pad_h, pad_h + img_h):
        for j in range(pad_w, pad_w + img_w):
            tmp = padded[i - pad_h : i + pad_h + 1, j - pad_w : j + pad_w + 1]
            output[i-pad_h, j-pad_w] = np.sum(tmp * kernel)
    return output


def part1(file):
    path = './HW3/test_picture/'
    laplacian_kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])
    laplacian_kernel2 = np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1]
    ])
    img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
    img = np.float32(img)
    
    img_gray_sharpening = convolve2d(img, laplacian_kernel)
    img_gray_sharpening = img - img_gray_sharpening
    img_gray_sharpening = np.clip(img_gray_sharpening, 0, 255).astype(np.uint8)
    result_file = 'spatial_laplacian1_' + file
    cv2.imwrite(os.path.join('./HW3', result_file), img_gray_sharpening)

    img_gray_sharpening = convolve2d(img, laplacian_kernel2)
    img_gray_sharpening = img - img_gray_sharpening
    img_gray_sharpening = np.clip(img_gray_sharpening, 0, 255).astype(np.uint8)
    result_file = 'spatial_laplacian2_' + file
    cv2.imwrite(os.path.join('./HW3', result_file), img_gray_sharpening)

def part2(k, file):
    path = './HW3/test_picture/'
    img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
    img_float = np.float64(img)
    row, col = img.shape
    crow, ccol = row // 2, col // 2
    # times(-1^(x+y)) to shift the low frequency to the center
    for i in range(row):
        for j in range(col):
            img_float[i, j] *= ((-1) ** (i + j))
    # Perform FFT
    f = np.fft.fft2(img_float)
    # construct laplacian filter in frequency domain
    laplacian_filter = np.zeros((row, col), dtype=np.float64)
    for u in range(row):
        for v in range(col):
            laplacian_filter[u, v] = -4 * np.pi**2 * ((u - crow) ** 2 + (v - ccol) ** 2)
    # bit-wise multiplication in frequency domain
    f_laplacian = f * laplacian_filter
    # Perform inverse FFT
    img_back = np.fft.ifft2(f_laplacian)
    img_back = np.real(img_back)
    # times(-1^(x+y)) to shift the low frequency to the center
    for i in range(row):
        for j in range(col):
            img_back[i, j] *= ((-1) ** (i + j))
    
    print(np.max(img_back))
    print(np.min(img_back))
    sharpened_img = img - k * img_back
    sharpened_img = np.clip(sharpened_img, 0, 255).astype(np.uint8)
    result_file = 'laplacian_frequency_' + file
    cv2.imwrite(os.path.join('./HW3', result_file), sharpened_img)

part2(8e-6, 'woman.jpg')
part2(5e-6, 'taj.jpg')
part1('woman.jpg')
part1('taj.jpg')