import numpy as np
import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture("/home/veerav/PycharmProjects/fuzzy/Bolt/img/%04d.jpg")

 # take first frame of the video
ret,frame = cap.read()
refPt = []
cropping = False
### to make a box to select roi using mouse

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



hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)          # convert BGR to HSV

roi_hist = cv2.calcHist([hsv_roi],[0],None,[180],[0,180]) # To find histogram of 1 channel i.e H
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)    # normalize histogram

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt for mean shift
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1 )

while(1):
    ret ,frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)   #to find confidence map
        #plt.imshow(dst)
        #plt.show()
 # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)  # to seek maximum of confidence map

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