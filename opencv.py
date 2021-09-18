import cv2
import numpy as np


#%%

img = cv2.imread('eunji.png')
# img = np.zeros((512,512,3), np.uint8)

print(img.shape)
img[:] = 0,0,0

cv2.line(img,(0,0), (img.shape[1],img.shape[0]), (0,255,0), 3)
cv2.rectangle(img, (0,0), (250,350),  (0,0,255), 3)
cv2.circle(img, (300,50), (30), (255,255,0), 5)
cv2.putText(img, " OPENCV ", (300, 100), cv2.FONT_HERSHEY_COMPLEX,1,(0,150,0), 1)

# cv2.imshow("Image", img)

cv2.waitKey(1)

#%%
#WARPED PERSPECTIVES

import cv2
import numpy as np

img = cv2.imread('cards.png')

print(img.shape)

width, height = 250, 350

pts1 = np.float32([[216, 670],[486, 612],[154, 305],[401, 259]])
pts2 = np.float32([[0,0],[width, 0],[0,height],[width,height]])

matrix = cv2.getPerspectiveTransform(pts1, pts2)

imgOutput = cv2.warpPerspective(img, matrix, (width, height))



cv2.imshow("Image", img)
cv2.imshow("Warped Image", imgOutput)


cv2.waitKey(1)


#%%
#JOINING IMAGES

img = cv2.imread('eunji.png')



imgHor = np.hstack((img, img))
imgVer = np.vstack((img, img))

cv2.imshow("Horizontal", imgHor)
cv2.imshow("Vertical", imgVer)

cv2.waitKey(1)

#%%
def empty():
    pass

# img = cv2.imread('eunji.png')

cv2.namedWindow('TrackBars')
cv2.resizeWindow('TrackBars', 640, 240)

cv2.createTrackbar('Hue Min', 'Trackbars',0,179, empty)
cv2.createTrackbar('Hue Max', 'Trackbars',0,179, empty)
cv2.createTrackbar('Sat Min', 'Trackbars',0,255, empty)
cv2.createTrackbar('Sat Max', 'Trackbars',255,255, empty)
cv2.createTrackbar('Value Min', 'Trackbars',0,255, empty)
cv2.createTrackbar('Value Max', 'Trackbars',255,255 , empty)


while True:
# img = cv2.imread('eunji.png')
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min", 'Trackbars')
    print(h_min)


    cv2.imshow("Image", img)
    cv2.imshow("HSV", imgHSV)

    cv2.waitKey(2)

#%%
#CONTOUR / SHAPE DETECTION

img = cv2.imread('shapes2.png')
scale_percent = 100
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

imgContour = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7,7), 1)

imgCanny = cv2.Canny(imgBlur, 50, 50)

imgBlack = np.zeros_like(img)

def getContours(img):
    contours, Hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print("Area is: ", area)
        # cv2.drawContours(imgContour, cnt, -1, (250, 0,0),1)
        if area > 50:
            cv2.drawContours(imgContour, cnt, -1, (250, 0,0),1)
            peri = cv2.arcLength(cnt, True)
            print("Perimeter is: ", peri)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            print(len(approx))
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            
            
            if objCor == 3:
                objectType = "Tri"
                
            elif objCor == 4:
                if (w/h < 1.03 and w/h > 0.97 ):
                    objectType = "Square"
                else:
                    objectType = "Rectangle"
                    
            elif objCor == 4:
                if (w/h < 1.03 and w/h > 0.97 ):
                    objectType = "Square"     
            else: objectType = "Polygon"

            cv2.rectangle(imgContour, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(imgContour, objectType, 
                        ((x+(w//2)-10), y+(h//2)-10), cv2.FONT_HERSHEY_COMPLEX,
                        0.4, (0,0,0), 1)



getContours(imgCanny)

    
cv2.imshow("Image", img)
cv2.imshow("Image Contours", imgContour)
cv2.imshow("Gray", imgGray)
cv2.imshow("Blur", imgBlur)
cv2.imshow("Canny", imgCanny)
cv2.imshow("Black", imgBlack)

cv2.waitKey(1)



#%%
#FACE DETECTION

import cv2

faceCascade = cv2.CascadeClassifier("")














