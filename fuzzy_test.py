from __future__ import division, print_function
import numpy as np
import cv2
import skfuzzy as fuzz
import matplotlib.pyplot as plt

cap = cv2.VideoCapture("/home/veerav/PycharmProjects/fuzzy/Badminton_ce2/img/%04d.jpg")

 # take first frame of the video
ret,frame = cap.read()
refPt = []
cropping = False
print(frame.shape)


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
# from teh image and display it
if len(refPt) == 2:
    roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
    cv2.imshow("ROI", roi)
    cv2.waitKey(0)

# close all open windows
cv2.destroyAllWindows()
track_window = (refPt[0][0],refPt[0][1],refPt[1][0]-refPt[0][0],refPt[1][1]-refPt[0][1])
 # setup initial location of window
#r,h,c,w = 250,90,400,125  # simply hardcoded the values
#track_window = (c,r,w,h)

 # set up the ROI for tracking
#roi = frame[r:r+h, c:c+w]

"""
========================
Fuzzy c-means clustering
========================

"""

# Generate test data

roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
print ('shape of image is  ',roi.shape)
xpts = np.zeros(roi.shape[0]*roi.shape[1])
ypts = np.zeros(roi.shape[0]*roi.shape[1])
#zpts = np.zeros(roi.shape[0]*roi.shape[1])
m = 0
for y in range(0,roi.shape[0]):
    for x in range(0,roi.shape[1]):
        xpts[m] = roi[y][x][0]
        #ypts[m] = roi[y][x][1]
        #zpts[m] = roi[y][x][2]
        m = m+1
alldata = xpts.reshape(1, roi.shape[0] * roi.shape[1])
#alldata = np.vstack((xpts,ypts,zpts))
#alldata = np.vstack((xpts,ypts))
print ('alldata shape is  ',alldata.shape)



#Clustering
#----------


cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(alldata, 8, 2, error=0.005, maxiter=100, init=None)

#print ('membership is ',u)

#print ('shape of partition matrix is   ', u.shape)

#print ('fp is  ',fpc)


cluster_assign = np.argmax(u,axis =0)  # Hardening for visualization
cluster_assign = cluster_assign.astype(np.uint8)
#print (cluster_assign.shape)
#print (type(cluster_assign))
roi_hist = cv2.calcHist([cluster_assign],[0],None,[8],[0,8])





#############
#hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
#roi_hist = cv2.calcHist([hsv_roi],[0,1],None,[32,32],[0,180,0,256])
#cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
#cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
cv2.normalize(roi_hist,roi_hist,alpha = 0,beta = 1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    ret ,frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,8],1)


 # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

 # Draw it on image
        x,y,w,h = track_window
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
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