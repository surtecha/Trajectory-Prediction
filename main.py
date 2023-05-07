import cv2
import cvzone
from cvzone.ColorModule import ColorFinder

cap = cv2.VideoCapture('Videos/vid2.mp4')

myColorFinder = ColorFinder(False)
hsvVals = {'hmin': 8, 'smin': 96, 'vmin': 115, 'hmax': 14, 'smax': 255, 'vmax': 255}

posList = []

while True:
    success, img = cap.read()
    #img = cv2.imread("Ball.png")
    img = img[0:900, :]

    #Find ball color
    imgColor, mask = myColorFinder.update(img, hsvVals)

    #Find ball location
    imgContours, contours = cvzone.findContours(img, mask, minArea=500)

    if contours:
        posList.append(contours[0]['center'])

    for i, pos in enumerate(posList):
        cv2.circle(imgContours, pos, 7, (0, 255, 0), cv2.FILLED)
        if i==0:
            cv2.line(imgContours, pos, pos, (0, 255, 0), 2)
        else:
            cv2.line(imgContours, pos, posList[i-1], (0, 255, 0), 2)

    # img = cv2.resize(img, (0, 0), fx=0.6, fy=0.6)
    # imgColor = cv2.resize(imgColor, (0, 0), fx=0.6, fy=0.6)
    imgContours = cv2.resize(imgContours, (0, 0), fx=0.6, fy=0.6)

    # cv2.imshow("Image", img)
    cv2.imshow("Image Color", imgContours)
    cv2.waitKey(100)