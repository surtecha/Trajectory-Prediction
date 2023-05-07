import cv2
import cvzone
import numpy as np
from cvzone.ColorModule import ColorFinder

cap = cv2.VideoCapture('Videos/vid2.mp4')

myColorFinder = ColorFinder(False)
hsvVals = {'hmin': 8, 'smin': 96, 'vmin': 115, 'hmax': 14, 'smax': 255, 'vmax': 255}

posListX, posListY = [], []
xList = [item for item in range(0, 1300)]

while True:
    success, img = cap.read()
    #img = cv2.imread("Ball.png")
    img = img[0:900, :]

    #Find ball color
    imgColor, mask = myColorFinder.update(img, hsvVals)

    #Find ball location
    imgContours, contours = cvzone.findContours(img, mask, minArea=500)

    if contours:
        posListX.append(contours[0]['center'][0])
        posListY.append(contours[0]['center'][1])

    if posListX:
        # Poly Regression: y = Ax^2 + Bx + c
        # Finding coefficients
        A, B, C = np.polyfit(posListX, posListY, 2)

        for i, (posX, posY) in enumerate(zip(posListX, posListY)):
            pos = (posX, posY)
            cv2.circle(imgContours, pos, 10, (0, 255, 0), cv2.FILLED)
            if i==0:
                cv2.line(imgContours, pos, pos, (0, 255, 0), 5)
            else:
                cv2.line(imgContours, pos, (posListX[i-1], posListY[i-1]), (0, 255, 0), 5)

        for x in xList:
            y = int(A*x**2 + B*x + C)
            cv2.circle(imgContours, (x, y), 2, (255, 0, 255), cv2.FILLED)


    # img = cv2.resize(img, (0, 0), fx=0.6, fy=0.6)
    # imgColor = cv2.resize(imgColor, (0, 0), fx=0.6, fy=0.6)
    imgContours = cv2.resize(imgContours, (0, 0), fx=0.6, fy=0.6)

    # cv2.imshow("Image", img)
    cv2.imshow("Image Color", imgContours)
    cv2.waitKey(70)