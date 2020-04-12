import os
import re

import pyautogui
import pytesseract
from PIL import Image

CONFIDENCE = 0.8


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


get_slot_region(how=2)
print(actual_value())
