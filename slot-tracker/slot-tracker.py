import os
import re

import pyautogui
import pytesseract
from PIL import Image

CONFIDENCE = 0.9


def data_path(filename):
    """A shortcut for joining the 'data/'' file path, since it is used so often. Returns the filename with 'data/' prepended."""
    return os.path.join("..", "data", filename)


def get_slot_region(how=0):
    """Obtains the region that the Slot is on the screen and assigns it to SLOT_REGION."""
    global SLOT_REGION
    if how == 0:
        # identify the top-left and top-right corner
        # logging.info('Finding slot region...')
        print("Finding slot region...")
        corner_top_left = pyautogui.locateOnScreen(
            data_path("corner_top_left.png"), confidence=CONFIDENCE
        )
        corner_top_right = pyautogui.locateOnScreen(
            data_path("corner_top_right.png"), confidence=CONFIDENCE
        )
        if (corner_top_left is None) or (corner_top_right is None):
            raise Exception("Could not find slot on screen. Is the slot visible?")

        # calculate the region of the entire slot
        topRightX = corner_top_right[0] + corner_top_right[2]  # left + width

        left_region = corner_top_left[0]
        top_region = corner_top_left[1]
        width_region = topRightX - corner_top_left[0]
        height_region = int(width_region * (204 / 1920))  # my bar is 1920 x 204
    elif how == 1:
        # identify the top-left and top-right corner
        # logging.info('Finding slot region...')
        print("Finding slot region...")
        corner_top_left = pyautogui.locateOnScreen(
            data_path("corner_top_left_long.png"), confidence=CONFIDENCE
        )
        corner_top_right = pyautogui.locateOnScreen(
            data_path("corner_top_right_long.png"), confidence=CONFIDENCE
        )
        if (corner_top_left is None) or (corner_top_right is None):
            raise Exception("Could not find slot on screen. Is the slot visible?")

        # calculate the region of the entire slot
        topRightX = corner_top_right[0] + corner_top_right[2]  # left + width

        left_region = corner_top_left[0]
        top_region = int(corner_top_left[1] + corner_top_left[3] / 1.5)
        width_region = topRightX - left_region
        height_region = int(width_region * (204 / 1920))  # my bar is 1920 x 204
    elif how == 2:
        # identify the bottom-left and bottom-right corner
        # logging.info('Finding slot region...')
        print("Finding slot region...")
        corner_bottom_left = pyautogui.locateOnScreen(
            data_path("corner_bottom_left.png"), confidence=CONFIDENCE
        )
        corner_bottom_right = pyautogui.locateOnScreen(
            data_path("corner_bottom_right.png"), confidence=CONFIDENCE
        )
        if (corner_bottom_left is None) or (corner_bottom_right is None):
            raise Exception("Could not find slot on screen. Is the slot visible?")

        # calculate the region of the entire slot
        topRightX = corner_bottom_right[0] + corner_bottom_right[2]  # left + width

        left_region = corner_bottom_left[0]
        width_region = topRightX - left_region
        top_region = int(corner_bottom_left[1] - (width_region * (159 / 1920)))
        height_region = int(width_region * (204 / 1920))  # my bar is 1920 x 204

    SLOT_REGION = (
        left_region,
        top_region,
        width_region,
        height_region,
    )
    # logging.info(f"Slot region found: {SLOT_REGION}")
    print(f"Slot region found: {SLOT_REGION}")


def actual_value(img=None):
    if img is None:
        img = pyautogui.screenshot(region=SLOT_REGION)
        img.show()
    window = pytesseract.image_to_string(img, lang="ita")
    denaro = re.findall(r"\bDenaro: \€\d+[.,]\d+\b", window)
    try:
        value_gross = denaro[0].split("€")[-1].replace(",", ".")
    except IndexError:
        value_gross = None
    return value_gross


# get_slot_region(how=2)
# print(actual_value())

################################
# Python program to illustrate
# multiscaling in template matching
import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt

# # Read the main image
# img_rgb = cv2.imread(data_path("slot_gd.png"))
# take a screenshot of the screen and store it in memory, then
# convert the PIL/Pillow image to an OpenCV compatible NumPy array
img_rgb = pyautogui.screenshot()
img_rgb = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)

# Convert to grayscale
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

# Read the template
template = cv2.imread(data_path("template.png"))
# Convert to grayscale
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
# Detect edges
template = cv2.Canny(template, 50, 200)
# Store width and height of template in w and h
w, h = template.shape[::-1]
(tH, tW) = template.shape[:2]

found = None

for scale in np.linspace(0.2, 4.0, 100)[::-1]:
    print(scale)
    # resize the image according to the scale, and keep track
    # of the ratio of the resizing
    resized = imutils.resize(img_gray, width=int(img_gray.shape[1] * scale))
    r = img_gray.shape[1] / float(resized.shape[1])

    # if the resized image is smaller than the template, then break
    # from the loop
    if resized.shape[0] < tH or resized.shape[1] < tW:
        break
    # detect edges in the resized, grayscale image and apply template
    # matching to find the template in the image
    edged = cv2.Canny(resized, 50, 200)
    result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

    # if we have found a new maximum correlation value, then update the bookkeeping variable
    if found is None or maxVal > found[0]:
        found = (maxVal, maxLoc, r)

# unpack the found varaible and compute the (x, y) coordinates
# of the bounding box based on the resized ratio
(_, maxLoc, r) = found
(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

# draw a bounding box around the detected result and display the image
plt.figure(1, figsize=(13, 10))
plt.imshow(cv2.rectangle(img_rgb, (startX, startY), (endX, endY), (0, 0, 255), 2))
plt.show()
