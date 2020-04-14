import datetime as dt
import os
import re
import time

import imutils
import numpy as np
import pyautogui
import pytesseract
from matplotlib import pyplot as plt
from PIL import Image, ImageGrab

import cv2
from slot-tracker import slot_classes

################################################################
# Parameters
################################################################

REGION_W = 1920
REGION_H = 220
MONEY_H = 45
TEMPLATE = "template.png"
BUTTONS = {
    "PLAY": (int(REGION_W / 2), int(REGION_H / 2)),
    "LIVELLO_DOWN": (int(REGION_W / 2 - 500), int(REGION_H / 2)),
    "LIVELLO_UP": (int(REGION_W / 2 - 330), int(REGION_H / 2)),
    "VALORE_DOWN": (int(REGION_W / 2 + 330), int(REGION_H / 2)),
    "VALORE_UP": (int(REGION_W / 2 + 540), int(REGION_H / 2)),
}
STATUS = False

################################################################
# Functions
################################################################


def data_path(filename):
    """A shortcut for joining the 'data/'' file path, since it is used so often. Returns the filename with 'data/' prepended."""
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", filename)
    )


def get_slot_region(which_resize="template"):
    """Obtains the region that the Slot is on the screen and assigns it to SLOT_REGION."""
    global SLOT_REGION, MONEY_REGION
    print("Start searching for the Slot region")

    if which_resize == "template":
        # take a screenshot of the screen and store it in memory, then
        # convert the PIL/Pillow image to an OpenCV compatible NumPy array
        img_rgb = pyautogui.screenshot()
        img_rgb = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
        # Convert to grayscale
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        # Detect edges
        img_edges = cv2.Canny(img_gray, 50, 200)
        # Store width and height of image
        (iH, iW) = img_edges.shape[:2]

        # Read the template
        template = cv2.imread(data_path(TEMPLATE))
        # Convert to grayscale
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        found = (np.NINF, None, None)

        for scale in np.linspace(0.1, 1.5, 100):
            # resize the image according to the scale, and keep track of the ratio of the resizing
            resized = imutils.resize(template, width=int(template.shape[1] * scale))
            r = template.shape[1] / float(resized.shape[1])

            # if the resized template is larger than the image, then break from the loop
            if resized.shape[0] > iH or resized.shape[1] > iW:
                break
            # detect edges in the resized, grayscale image and apply template matching to find the template in the image
            edged = cv2.Canny(resized, 50, 200)
            result = cv2.matchTemplate(img_edges, edged, cv2.TM_CCOEFF)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

            # if we have found a new maximum correlation value, then update the bookkeeping variable
            if maxVal > found[0]:
                found = (maxVal, maxLoc, r)

        # unpack the found varaible and compute the (x, y) coordinates
        # of the bounding box based on the resized ratio
        (_, maxLoc, r) = found
        (startX, startY) = (int(maxLoc[0]), int(maxLoc[1]))
        (endX, endY) = (
            int(maxLoc[0] + (template.shape[1] / r)),
            int(maxLoc[1] + ((REGION_H * template.shape[1] / REGION_W) / r)),
        )

    elif which_resize == "screen":
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
        # Store width and height of template
        (tH, tW) = template.shape[:2]

        found = (np.NINF, None, None)

        for scale in np.linspace(0.2, 3.0, 50)[::-1]:
            # resize the image according to the scale, and keep track of the ratio of the resizing
            resized = imutils.resize(img_gray, width=int(img_gray.shape[1] * scale))
            r = img_gray.shape[1] / float(resized.shape[1])

            # if the resized image is smaller than the template, then break from the loop
            if resized.shape[0] < tH or resized.shape[1] < tW:
                break
            # detect edges in the resized, grayscale image and apply template
            # matching to find the template in the image
            edged = cv2.Canny(resized, 50, 200)
            result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

            # if we have found a new maximum correlation value, then update the bookkeeping variable
            if maxVal > found[0]:
                found = (maxVal, maxLoc, r)

        # unpack the found varaible and compute the (x, y) coordinates
        # of the bounding box based on the resized ratio
        (_, maxLoc, r) = found
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (
            int((maxLoc[0] + tW) * r),
            int((maxLoc[1] + (REGION_H * tW / REGION_W)) * r),
        )

    else:
        raise Exception("Invalid which_resize")

    SLOT_REGION = (
        startX,
        startY,
        endX - startX,  # width
        endY - startY,  # height
    )

    money_h = int(MONEY_H * SLOT_REGION[2] / REGION_W)

    MONEY_REGION = (
        startX,
        endY - money_h,
        endX - startX,  # width
        money_h,  # height
    )

    # logging.info(f"Slot region found: {SLOT_REGION}")
    print(f"Slot region found: {SLOT_REGION}")


def plot_slot_region(figsize=(15, 3)):
    plt.figure(figsize=figsize)
    plt.imshow(pyautogui.screenshot(region=SLOT_REGION))
    plt.axis("off")
    plt.show()


def update_slot_buttons():
    BUTTONS["PLAY"] = (
        int(SLOT_REGION[0] + SLOT_REGION[2] / 2),
        int(SLOT_REGION[1] + SLOT_REGION[3] / 2),
    )
    BUTTONS["LIVELLO_DOWN"] = (
        int(BUTTONS["PLAY"][0] - (500 * SLOT_REGION[2] / REGION_W)),
        BUTTONS["PLAY"][1],
    )
    BUTTONS["LIVELLO_UP"] = (
        int(BUTTONS["PLAY"][0] - (330 * SLOT_REGION[2] / REGION_W)),
        BUTTONS["PLAY"][1],
    )
    BUTTONS["VALORE_DOWN"] = (
        int(BUTTONS["PLAY"][0] + (330 * SLOT_REGION[2] / REGION_W)),
        BUTTONS["PLAY"][1],
    )
    BUTTONS["VALORE_UP"] = (
        int(BUTTONS["PLAY"][0] + (540 * SLOT_REGION[2] / REGION_W)),
        BUTTONS["PLAY"][1],
    )


def test_slot_buttons():
    print("Starting test buttons")
    for key, value in BUTTONS.items():
        print(key)
        pyautogui.moveTo(value)
        time.sleep(2)


def actual_value():
    img = pyautogui.screenshot(region=MONEY_REGION)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY_INV)[1]
    img = cv2.bitwise_not(img)
    window = pytesseract.image_to_string(img, lang="ita", config="--psm 7 --oem 3")
    money_str = re.findall(
        r"\b(?:Denaro:|Credito:|S*aldo:|\€|) \€*(\d+[.,]*\d+)\b",
        window,
        flags=re.IGNORECASE,
    )
    try:
        value_gross = float(re.sub(r"[,.]", "", money_str[0])) / 100
    except IndexError:
        value_gross = np.nan
    return value_gross


def get_slot_status(COLOR):
    global STATUS
    STATUS = ImageGrab.grab().getpixel(BUTTONS["PLAY"]) == COLOR
    return STATUS


def main():
    get_slot_region(which_resize="screen")
    plot_slot_region()
    update_slot_buttons()
    test_slot_buttons()
    # Initialize COLOR
    print("Taking baseline color")
    COLOR = ImageGrab.grab().getpixel(BUTTONS["PLAY"])
    money = actual_value()
    print(f"Saldo rilevato: {money}")
    confirm = pyautogui.confirm(
        text="Is it all good?", title="START", buttons=["Yes", "No"]
    )
    if confirm == "No":
        pass
    elif confirm == "Yes":
        ROLLOVER = pyautogui.prompt(
            text="How much Rollover?", title="Rollover", default=0
        )
        if ROLLOVER is None or ROLLOVER < 0:
            pass
        else:
            bet = 1
            total = 0

            while total < ROLLOVER:
                # get current time
                timestamp = dt.datetime.now()

                # play
                pyautogui.click(BUTTONS["PLAY"])

                # ready
                while not STATUS:
                    get_slot_status(COLOR)

                # gain

                # plot

                # ML

                # update bet and total


if __name__ == "__main__":
    main()
