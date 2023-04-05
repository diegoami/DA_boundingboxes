import cv2
import pytesseract
import numpy as np
import imutils
import matplotlib.pyplot as plt


def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply thresholding (use cv2.THRESH_BINARY instead of cv2.THRESH_BINARY_INV)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return image, thresh

def filter_contours(contours):
    filtered_contours = []
    for contour in contours:
        # Compute the bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        # Define aspect ratio and area filtering criteria
        aspect_ratio = w / float(h)
        min_area = 2
        max_aspect_ratio = 200
        if w * h > min_area and 1/ max_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
            filtered_contours.append((x, y, w, h))
    return filtered_contours

def draw_bounding_boxes(image, bounding_boxes):
    for x, y, w, h in bounding_boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

def main(image_path):
    image, thresh = preprocess_image(image_path)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = filter_contours(contours)
    image_with_boxes = draw_bounding_boxes(image, bounding_boxes)
    cv2.imshow("Text Detection", image_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "image1.jpg"
    main(image_path)
