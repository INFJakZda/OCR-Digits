import glob
import os

import cv2


def process(img):
    # TODO:Implement sth
    pass


def main(path):
    for f_name in os.listdir(path):
        # print(os.path.exists())
        img = cv2.imread(os.path.join(path, f_name))
        process(img)


if __name__ == "__main__":
    path = os.path.join(".", "data")
    main(path)

