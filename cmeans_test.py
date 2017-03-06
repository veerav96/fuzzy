"""
========================
Fuzzy c-means clustering
========================

Fuzzy logic principles can be used to cluster multidimensional data, assigning
each point a *membership* in each cluster center from 0 to 100 percent. This
can be very powerful compared to traditional hard-thresholded clustering where
every point is assigned a crisp, exact label.

Fuzzy c-means clustering is accomplished via ``skfuzzy.cmeans``, and the
output from this function can be repurposed to classify new data according to
the calculated clusters (also known as *prediction*) via
``skfuzzy.cmeans_predict``

Data generation and setup
-------------------------

In this example we will first undertake necessary imports, then define some
test data to work with.

"""
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import cv2


# Define three cluster centers
#centers = [[4, 2],
#           [1, 7],
#           [5, 6]]



# Generate test data

image = cv2.imread('a.jpg')
print ('shape of image is  ',image.shape)
xpts = np.zeros(image.shape[0]*image.shape[1])

print('shape of xpts ',xpts.shape)
ypts = np.zeros(image.shape[0]*image.shape[1])

zpts = np.zeros(image.shape[0]*image.shape[1])

m = 0
for y in range(0,image.shape[0]):
    for x in range(0,image.shape[1]):
        xpts[m] = image[y][x][0]
        #ypts[m] = image[y][x][1]
        #zpts[m] = image[y][x][2]
        m = m+1
print('xpts is ',xpts)
print('xpts shape',xpts.shape)
#alldata = np.vstack((xpts,ypts,zpts))
#alldata = np.vstack((xpts,ypts))
alldata = xpts.reshape(1,image.shape[0]*image.shape[1])

print ('alldata shape is  ',alldata.shape)
"""


Clustering
----------

Above is our test data. We see three distinct blobs. However, what would happen
if we didn't know how many clusters we should expect? Perhaps if the data were
not so clearly clustered?

Let's try clustering our data several times, with between 2 and 9 clusters.

"""

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(alldata, 8, 2, error=0.005, maxiter=50, init=None)

print ('membership is ',u)


print ('shape of partition matrix is   ', u.shape )


cluster_assign = np.argmax(u,axis =0)  # Hardening for visualization
cluster_assign = cluster_assign.reshape(1,image.shape[0]*image.shape[1])
cluster_assign = cluster_assign.astype(np.uint8)
print ('shape of cluster_assign is ',cluster_assign.shape)
print ('cluster_assign is ',cluster_assign)
hist = cv2.calcHist([cluster_assign],[0],None,[8],[0,8])
print (hist.shape)

hist = np.sort(hist,axis =0)

print('hist is ',hist)




