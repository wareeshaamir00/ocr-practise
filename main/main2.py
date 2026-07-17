import cv2
import numpy as np

# Read the image
im = cv2.imread("images/image.webp")

# Convert image from BGR to Grayscale
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Remove noise from the image
def noise_removal(image):
    kernel = np.ones((5, 5), np.uint8)

    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)

    return image


# ---------- Processing Pipeline ----------

# Convert to grayscale
gray_image = grayscale(im)

# Save and display grayscale image
cv2.imwrite("images/image_gray.webp", gray_image)
cv2.imshow("Grayscale Image", gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert to black and white (binarization)
_, im_bw = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)

# Save and display binary image
cv2.imwrite("images/image_bw.webp", im_bw)
cv2.imshow("Binary Image", im_bw)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Remove noise
no_noise = noise_removal(im_bw)

# Save and display cleaned image
cv2.imwrite("images/image_no_noise.webp", no_noise)
cv2.imshow("Noise Removed Image", no_noise)
cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2
import numpy as np

def deskew(image):
    # Find coordinates of all non-zero (white) pixels
    coords = np.column_stack(np.where(image > 0))

    # Find the angle of the minimum area rectangle
    angle = cv2.minAreaRect(coords)[-1]

    # Correct the angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Get image dimensions
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Create rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Rotate the image
    deskewed = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return deskewed
# Remove noise
no_noise = noise_removal(im_bw)

# Deskew the image
deskewed = deskew(no_noise)

# Save and display
cv2.imwrite("images/image_deskew.webp", deskewed)
cv2.imshow("Deskewed Image", deskewed)
cv2.waitKey(0)
cv2.destroyAllWindows()