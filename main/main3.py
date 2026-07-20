import cv2
import numpy as np
im = cv2.imread("images/page_01.jpg")
gs = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
def binarise(gs):
    return cv2.threshold(gs, 210, 230, cv2.THRESH_BINARY)[1]
grayscale = binarise(gs)
def noise(grayscale):
    kernel = np.ones((1, 1), np.uint8)
    grayscale = cv2.dilate(grayscale, kernel, iterations=2)
    grayscale = cv2.erode(grayscale, kernel, iterations=2)
    grayscale = cv2.morphologyEx(grayscale, cv2.MORPH_OPEN, kernel)
    grayscale = cv2.medianBlur(grayscale, 3)
    return grayscale
def thick(grayscale):
    k = np.ones((1, 1), np.uint8)
    grayscale = cv2.dilate(grayscale, k, iterations=1)
    return grayscale
def preprocess():
    return thick(noise(binarise(gs)))
