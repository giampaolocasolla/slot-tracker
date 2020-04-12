import re
import os

import pyautogui
import pytesseract

try:
    from PIL import Image
except ImportError:
    import Image
try:
    from PIL import ImageGrab
except ImportError:
    import ImageGrab


def data_path(filename):
    """A shortcut for joining the 'data/'' file path, since it is used so often. Returns the filename with 'data/' prepended."""
    return os.path.join("..", "data", filename)


def get_slot_region():
    """Obtains the region that the Slot is on the screen and assigns it to SLOT_REGION."""
    global SLOT_REGION

    # identify the top-left and top-right corner
    # logging.info('Finding slot region...')
    print("Finding slot region...")
    corner_top_left = pyautogui.locateOnScreen(
        data_path("corner_top_left.png"), confidence=0.9
    )
    corner_top_right = pyautogui.locateOnScreen(
        data_path("corner_top_right.png"), confidence=0.9
    )
    if (corner_top_left is None) or (corner_top_right is None):
        raise Exception("Could not find slot on screen. Is the slot visible?")

    # calculate the region of the entire slot
    topRightX = corner_top_right[0] + corner_top_right[2]  # left + width

    left_region = corner_top_left[0]
    top_region = corner_top_left[1]
    width_region = topRightX - corner_top_left[0]
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
        img = ImageGrab.grab(bbox=SLOT_REGION)
    window = pytesseract.image_to_string(img, lang="ita")
    denaro = re.findall(r"\bDenaro: \€\d+[.,]\d+\b", window)
    try:
        value_gross = denaro[0].split("€")[-1].replace(",", ".")
    except IndexError:
        value_gross = None
    return value_gross


get_slot_region()
print(actual_value())
