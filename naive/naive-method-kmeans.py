import numpy as np
import cv2

img = cv2.imread('images/test3.jpg')
shape = img.shape
cv2.imshow('ORIGINAL IMG', cv2.resize(img, (shape[1]//2, shape[0]//2)))

Z = img.reshape((-1,3))
# convert to np.float32
Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 11
ret, label, center=cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res = res.reshape(shape)

cv2.imshow("KMEANS",  cv2.resize(res, (shape[1]//2, shape[0]//2)))
cv2.waitKey(0)