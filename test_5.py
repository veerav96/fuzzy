import numpy as np
import cv2
x = ([1,2,3,6])
y = ([10,15,20,5])
print type(x)
a = np.hstack((x,y))
b = np.vstack((x,y))
print a
print type(a)
print b
print type(b)
print b.shape

