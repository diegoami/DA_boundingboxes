import cv2
import numpy as np

image = cv2.imread("image1.jpg")
mask = np.zeros(image.shape, dtype=np.uint8)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3, 3), 0)
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 9)

# Create horizontal kernel then dilate to connect text contours
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dilate = cv2.dilate(thresh, kernel, iterations=2)

# Find contours and filter out noise using contour approximation and area filtering
cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    x, y, w, h = cv2.boundingRect(c)
    area = w * h
    ar = w / float(h)
    if area > 1200 and area < 50000 and ar < 8:
        cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)

# Bitwise-and input image and mask to get result
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
result = cv2.bitwise_and(image, image, mask=mask)
result[mask == 0] = (255, 255, 255)  # Color background white

# NEW CODE HERE TO END _____________________________________________________________
gray2 = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
thresh2 = cv2.threshold(gray2, 128, 255, cv2.THRESH_BINARY)[1]
thresh2 = 255 - thresh2
kernel = np.ones((5, 191), np.uint8)
close = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)

# get external contours
contours = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

# draw contours
result2 = result.copy()
for cntr in contours:
    # get bounding boxes
    pad = 10
    x, y, w, h = cv2.boundingRect(cntr)
    cv2.rectangle(result2, (x - pad, y - pad), (x + w + pad, y + h + pad), (0, 0, 255), 4)

cv2.imwrite("john_bboxes.jpg", result2)

cv2.imshow("mask", mask)
cv2.imshow("thresh", thresh)
cv2.imshow("dilate", dilate)
cv2.imshow("result", result)
cv2.imshow("gray2", gray2)
cv2.imshow("thresh2", thresh2)
cv2.imshow("close", close)
cv2.imshow("result2", result2)

cv2.waitKey(0)
cv2.destroyAllWindows()