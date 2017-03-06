import cv2
import numpy as np

x = cv2.imread('a.jpg')
print type(x)
#print x


#b = np.array([[1,2,3],[4,5,6]])
#print b
#print b.shape
#print b[1][1]

c= np.array([ [ [1,2,3,4],[5,6,7,5] ],   [[11,12,13,9],[15,16,17,89]],    [[1,5,6,7],[84,3,5,9]]   ])
print c.shape
print c[1][1][3]

cv2.imshow('asdf',x)
cv2.waitKey(0)

