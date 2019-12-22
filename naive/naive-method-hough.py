import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('images/test3.jpg')
shape = img.shape
cv2.imshow('ORIGINAL IMG', cv2.resize(img, (shape[1]//2, shape[0]//2)))

gray = cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
cv2.imshow("NO BLUR",  cv2.resize(binary, (shape[1]//2, shape[0]//2)))
binary = cv2.medianBlur(binary, 5)
cv2.imshow("BLUR",  cv2.resize(binary, (shape[1]//2, shape[0]//2)))

circles = cv2.HoughCircles(binary, cv2.HOUGH_GRADIENT, 1, min(shape[0], shape[1])//10, param1=80, 
param2=20, minRadius=20, maxRadius=max(shape[0], shape[1])//10)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=3)
cv2.imshow("OPEN",  cv2.resize(binary, (shape[1]//2, shape[0]//2)))

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
      cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0),2)
      cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 2)
cv2.imshow("HOUGH",  cv2.resize(img, (shape[1]//2, shape[0]//2)))
cv2.waitKey(0)