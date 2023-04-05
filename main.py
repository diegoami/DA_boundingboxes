import cv2
import pytesseract
import numpy as np
import imutils
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('image1.jpg')

# Preprocess the image
#img = imutils.resize(img, width=600)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Find the contours in the image
contours = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

# Loop over the contours and extract the text
for contour in contours:
    (x, y, w, h) = cv2.boundingRect(contour)
    roi = gray[y:y + h, x:x + w]
    text = pytesseract.image_to_string(roi, config='--psm 11')

    # Draw the bounding box and text
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Display the final image
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
