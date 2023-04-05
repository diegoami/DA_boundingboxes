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

def filter_contours(thresh, contours, min_area=50, max_aspect_ratio=5, nested=False):
    filtered_contours = []
    for contour in contours:
        # Compute the bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        # Define aspect ratio and area filtering criteria
        aspect_ratio = w / float(h)

        if w * h > min_area and 1 / max_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
            print(f"Detected ({(x, y, w, h)})")
            filtered_contours.append((x, y, w, h))

            if nested:
                # Find inner contours by thresholding the region within the bounding rectangle
                roi = thresh[y:y + h, x:x + w]
                inner_contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                #print(inner_contours)
                inner_boxes = filter_contours(thresh, inner_contours, min_area=min_area, max_aspect_ratio=max_aspect_ratio, nested=False)
                # Adjust the coordinates of the inner bounding rectangles to be relative to the original image
                inner_boxes = [(x + inner_x, y + inner_y, inner_w, inner_h) for (inner_x, inner_y, inner_w, inner_h) in inner_boxes]
                filtered_contours.extend(inner_boxes)
        else:
            print(f"Ignoring ({(x, y, w, h)})")

    return filtered_contours

def draw_bounding_boxes(image, bounding_boxes):
    for x, y, w, h in bounding_boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

def main(image_path):
    image, thresh = preprocess_image(image_path)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = filter_contours(thresh, contours, 5, 20, True)
    print(bounding_boxes)
    image_with_boxes = draw_bounding_boxes(image, bounding_boxes)
    cv2.imshow("Text Detection", image_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "image1_trim.jpg"
    main(image_path)
