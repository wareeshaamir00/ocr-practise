import cv2
im = cv2.imread("images/image.webp")#openimage
def grayscale(image):#convert image from bgr to gray
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image = grayscale(im)
cv2.imwrite("images/image_gray.webp", gray_image)
cv2.imshow("image_gray",  gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
thresh, im_bw = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
cv2.imwrite("images/image_bw.webp", im_bw)
cv2.imshow("image_bw",  im_bw)
cv2.waitKey(0)
cv2.destroyAllWindows()
def noise_removal(image):#remove noise from image
    import numpy as np
    kernel = np.ones((5, 5), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return image
no_noise = noise_removal(im_bw)
cv2.imwrite("images/image_no_noise.webp", no_noise)
cv2.imshow("image_no_noise",  no_noise)
cv2.waitKey(0)
cv2.destroyAllWindows()