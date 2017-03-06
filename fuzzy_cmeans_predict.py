from __future__ import division, print_function
import numpy as np
import cv2
import skfuzzy as fuzz
import math
import matplotlib.pyplot as plt

cap = cv2.VideoCapture("/home/veerav/PycharmProjects/fuzzy/Badminton_ce2/img/%04d.jpg")

 # take first frame of the video
ret,frame = cap.read()
refPt = []
cropping = False



def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False

        # draw a rectangle around the region of interest
        cv2.rectangle(frame, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", frame)


# load the image, clone it, and setup the mouse callback function

clone = frame.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow("image", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        image = clone.copy()

    # if the 'c' key is pressed, break from the loop
    elif key == ord("c"):
        break

# if there are two reference points, then crop the region of interest
# from the image and display it
if len(refPt) == 2:
    roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]

    cv2.imshow("ROI", roi)
    cv2.waitKey(0)



# close all open windows
cv2.destroyAllWindows()
track_window = (refPt[0][0],refPt[0][1],refPt[1][0]-refPt[0][0],refPt[1][1]-refPt[0][1])


"""
========================
Fuzzy c-means clustering
========================

"""

# Generate test data for region of interest

roi = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
print ('shape of image is  ',roi.shape)
xpts = np.zeros(roi.shape[0]*roi.shape[1])
ypts = np.zeros(roi.shape[0]*roi.shape[1])
zpts = np.zeros(roi.shape[0]*roi.shape[1])

m = 0
for y in range(0,roi.shape[0]):
    for x in range(0,roi.shape[1]):
        xpts[m] = roi[y][x][0]                #channel 1 pixels
        #ypts[m] = roi[y][x][1]
        #zpts[m] = roi[y][x][2]
        m = m+1
#alldata = np.vstack((xpts,ypts))
#alldata = np.vstack((xpts,ypts,zpts))
alldata = xpts.reshape(1,roi.shape[0]*roi.shape[1])

print ('alldata shape is  ',alldata.shape)



#Clustering

# c-means clustering
#we obtain 8 clusters
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(alldata, 8, 2, error=0.05, maxiter=20, init=None)## u is fuzzy partition matrix

#print ('membership is ',u)

#print ('shape of partition matrix is   ', u.shape)

#print ('fp is  ',fpc)

# u is the fuzzy partition matrix
cluster_assign = np.argmax(u,axis =0)  # Hardening for visualization #to obtain maximum from each column
cluster_assign = cluster_assign.astype(np.uint8)
#print (cluster_assign.shape)
#print (type(cluster_assign))
roi_hist = cv2.calcHist([cluster_assign],[0],None,[8],[0,8])
roi_hist = np.sort(roi_hist,axis=0)
print('roi_hist is ',roi_hist)



#### clustering for neighbouring boxes,
#### comparing histograms for all and storing similarity value
#### use its index value to find new tracked window location

x2, y2, w, h = track_window


while(1):
    ret ,frame = cap.read()

    if ret == True:


        store_similarity = np.zeros((10, 10))


        i =0

        for y1 in range(y2 - 50, y2 + 50,10):
            j=0
            for x1 in range(x2 - 50, x2 +50,10):
                clone = frame.copy()
                roi = clone[y1:y1+h, x1:x1+w]
                roi = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
                m = 0
                for y in range(0, roi.shape[0]):
                    for x in range(0, roi.shape[1]):


                        xpts[m] = roi[y][x][0]  ##channel 1 pts
                        #ypts[m] = roi[y][x][1]
                        #zpts[m] = roi[y][x][2]
                        m = m + 1

                newdata = xpts.reshape(1, roi.shape[0] * roi.shape[1])
                #alldata = np.vstack((xpts, ypts))

                u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(newdata, cntr, 2, error=0.05, maxiter=20)
                #cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(alldata, 8, 2, error=0.05, maxiter=10, init=None)
                cluster_assign = np.argmax(u, axis=0)  # Hardening for visualization
                cluster_assign = cluster_assign.astype(np.uint8)
                # print (cluster_assign.shape)
                # print (type(cluster_assign))
                hist = cv2.calcHist([cluster_assign], [0], None, [8], [0, 8])
                hist = np.sort(hist,axis=0)

                #hist = cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                store_similarity[i][j] = cv2.compareHist(roi_hist, hist, cv2.HISTCMP_CORREL)

                #print ('y1 is ',y1)
                #print ('x1 is ',x1)
                j=j+1
            i = i+1
        print(store_similarity)
        i, j = np.unravel_index(store_similarity.argmax(), store_similarity.shape)
        print ('i is ',i)
        print ('j is ',j)
        x2 = x2+10*j-50
        y2 = y2+10*i-50

        #####


 # Draw it on image

        img2 = cv2.rectangle(frame, (x2,y2), (x2+w,y2+h), 255,2)
        cv2.imshow('img2',img2)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k)+".jpg",img2)

    else:
        break


cv2.destroyAllWindows()
cap.release()