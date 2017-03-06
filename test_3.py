import cv2
import numpy as np
import matplotlib.pyplot as plt



mat = np.zeros((10,10,3))
print type(mat)
for i in range(0,3):                                               #######image
    for y in range(0,10):
        for x in range(0,10):
            if i == 0:
                mat[y][x][i] =128
            elif i==1:
                mat[y][x][i] = 240
            else:
                 mat[y][x][i] = 140
print mat
image = mat
xpts = np.zeros(image.shape[0]*image.shape[1])
ypts = np.zeros(image.shape[0]*image.shape[1])
zpts = np.zeros(image.shape[0]*image.shape[1])                                  ####to read data
m = 0
for y in range(0,image.shape[0]):
    for x in range(0,image.shape[1]):

        xpts[m] = image[y][x][0]
        ypts[m] = image[y][x][1]
        zpts[m] = image[y][x][2]
        m = m+1
data = np.vstack((xpts,ypts,zpts))
print data

#cv2.imwrite('matt.jpg',mat)
#matt = cv2.imread('matt.jpg')

#plt.imshow(mat)
#plt.show()
#cv2.imshow('mat',matt)
#cv2.waitKey(0)