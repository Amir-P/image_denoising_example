import numpy as np
import math

def psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return 100
    return 20 * math.log10(255 / math.sqrt(mse))

def hardThresholding(image, threshold):
    pixelMax = np.max(image)
    if pixelMax > 1:
        pixelMax = 255
    else:
        pixelMax = 1
    mean = pixelMax / 2
    if image.ndim == 3:
        return [[value if abs(np.sum(value) - mean) > threshold else (np.zeros(len(value)) + mean) for value in row] for row in image]
    return [[value if abs(value - mean) > threshold else mean for value in row] for row in image]

def softThresholding(image, threshold):
    pixelMax = np.max(image)
    if pixelMax > 1:
        pixelMax = 255
    else:
        pixelMax = 1
    mean = pixelMax / 2
    if image.ndim == 3:
        return [[value - threshold if abs(np.sum(value) - mean) > threshold else (np.zeros(len(value)) + mean) for value in row] for row in image]
    return [[value - threshold if abs(value - mean) > threshold else mean for value in row] for row in image]
