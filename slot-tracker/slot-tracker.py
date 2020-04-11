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


print(actual_value(Image.open("../data/prova_slot.png")))
print(actual_value())
