import sys
import numpy as np
import matplotlib.pyplot as plt
import utils

def averaging(image, kernelSize=3, stride=1):
    denoisedImage = np.zeros(image.shape)
    kernelY = 0
    while kernelY <= image.shape[1] - kernelSize:
        kernelX = 0
        while kernelX <= image.shape[0] - kernelSize:
            denoisedImage[kernelX + int(kernelSize / 2), kernelY + int(kernelSize / 2)] = np.mean(image[kernelX:kernelX + kernelSize, kernelY:kernelY + kernelSize], axis=(0, 1))
            kernelX += stride
        kernelY += stride
    return denoisedImage

def median(image, kernelSize=3, stride=1):
    denoisedImage = np.zeros(image.shape)
    kernelY = 0
    while kernelY <= image.shape[1] - kernelSize:
        kernelX = 0
        while kernelX <= image.shape[0] - kernelSize:
            denoisedImage[kernelX + int(kernelSize / 2), kernelY + int(kernelSize / 2)] = np.asarray(sorted(image[kernelX:kernelX + kernelSize, kernelY:kernelY + kernelSize], key=lambda x: np.sum(x)))[int(kernelSize / 2), int(kernelSize / 2)]
            kernelX += stride
        kernelY += stride
    return denoisedImage

def thresholding(image, mode, threshold):
    if mode == 'hard':
        return utils.hardThresholding(image, threshold)
    elif mode == 'soft':
        return utils.softThresholding(image, threshold)
    return None