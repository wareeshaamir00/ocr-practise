import cv2
import numpy as np
im = cv2.imread("images/page_01.jpg")
gs = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
def binarise(gs):
    return cv2.threshold(gs, 210, 230, cv2.THRESH_BINARY_INV)[1]
grayscale = binarise(gs)
def noise(grayscale):
    kernel = np.ones((1, 1), np.uint8)
    grayscale = cv2.dilate(grayscale, kernel, iterations=2)
    grayscale = cv2.erode(grayscale, kernel, iterations=2)
    grayscale = cv2.morphologyEx(grayscale, cv2.MORPH_OPEN, kernel)
    grayscale = cv2.medianBlur(grayscale, 3)
    return grayscale
def contour(grayscale):
    kernel = np.ones((3, 15), np.uint8)
    binary = cv2.dilate(grayscale, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= 200:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return im
if im is None:
    print("Error: Could not read the image.")
    exit()
else:
    cv2.imshow("original", contour(noise(binarise(gs))))
    cv2.waitKey(0)
    cv2.imwrite("output/Images.jpg", contour(noise(binarise(gs))))
def preprocess():
    return noise(binarise(gs))