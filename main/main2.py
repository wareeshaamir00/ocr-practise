import cv2
import numpy as np
im = cv2.imread("images/image.webp")
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
def invert(image):
    return cv2.bitwise_not(image)
def blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)
def noise(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return image
def bw(image):
    return cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)[1]
def adaptive_threshold(image):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
def resize(image, width, height):
    return cv2.resize(image, (width, height))
def rotate(image):
    (h, w) = image.shape[:2]
    center = (w // 5, h // 5)
    m = cv2.getRotationMatrix2D(center, 90, 1)
    rotated = cv2.warpAffine(image, m, (w, h))
    return rotated
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 90, 1)
    deskewed = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return deskewed
cv2.imwrite("output/Images.webp", grayscale(im))
cv2.imshow("original", deskew(grayscale(im)))
cv2.waitKey(0)
cv2.destroyAllWindows()