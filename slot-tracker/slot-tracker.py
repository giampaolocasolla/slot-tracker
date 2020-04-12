import re

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
    return os.path.join("data", filename)


def actual_value(img=None):
    if img is None:
        img = ImageGrab.grab()
    window = pytesseract.image_to_string(img, lang="ita")
    denaro = re.findall(r"\bDenaro: \€\d+[.,]\d+\b", window)
    try:
        value_gross = denaro[0].split("€")[-1].replace(",", ".")
    except IndexError:
        value_gross = None
    return value_gross


# print(actual_value(Image.open("../data/slot_bar.png")))
# print(actual_value())
