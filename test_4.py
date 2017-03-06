import cv2
import numpy as np

image = cv2.imread('a.jpg')
xpts = np.zeros(image.shape[0]*image.shape[1])
ypts = np.zeros(image.shape[0]*image.shape[1])
zpts = np.zeros(image.shape[0]*image.shape[1])
m = 0
for y in range(0,image.shape[0]):
    for x in range(0,image.shape[1]):
        xpts[m] = image[y][x][0]
        ypts[m] = image[y][x][1]
        zpts[m] = image[y][x][2]
        m = m+1
data = np.vstack((xpts,ypts,zpts))
