import glob
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from worddetector import ocr


def main(path):
    for f_name in os.listdir(path):
        print(f_name)
        if f_name[-3:] == "jpg":
            ocr(os.path.join(path, f_name))

        # print(os.path.exists())


if __name__ == "__main__":
    path = os.path.join(".", "data")
    main(path)
