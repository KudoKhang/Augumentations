from Augs import Augment
import cv2

A = Augment()

img = cv2.imread('src/2.png')
mask = cv2.imread('src/mask_face.png')

i, m = A.Perspective(img, mask)

cv2.imshow('result', i)
cv2.imshow('mask', m)
cv2.waitKey(0)

