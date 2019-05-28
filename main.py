import glob
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt


def show_with_resize(winname, img, width=None):
    try:
        h, w, ch = img.shape
    except ValueError:
        h, w = img.shape
    if width is not None:
        ratio = h/w
        resized_img = cv2.resize(img, (width, int(ratio * width)))
        cv2.imshow(winname, resized_img)
    else:
        cv2.imshow(winname, img)


def apply_sobel(gray):
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)


def find_lines_mask(img):
    ksize_median_blur = 19
    kernel_size_dilate = (5, 5)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    show_with_resize("normal", gray, 450)

    grad = apply_sobel(gray)

    blur_grad = cv2.medianBlur(grad, ksize_median_blur)
    kernel = np.ones(kernel_size_dilate, np.uint8)
    # show_with_resize("grad", grad, 450)

    # erode = cv2.erode(blur_grad, kernel, iterations=3)
    dilate = cv2.dilate(blur_grad, kernel, iterations=5)
    # show_with_resize("erode", erode, 450)
    # show_with_resize("dilate", dilate, 450)

    mask = np.zeros(dilate.shape, dtype=np.uint8)
    lines_sum = np.sum(dilate, axis=1)
    lines_sum_mean = np.mean(lines_sum)
    lines_sum_med = np.median(lines_sum)
    print("median: {}, mean: {}".format(lines_sum_mean, lines_sum_med))
    for i, line_sum in enumerate(lines_sum):
        line = dilate[i]
        line_mean = np.mean(line)
        line_med = np.median(line)
        for j, pixel in enumerate(line):
            if pixel > line_med * 1.6 and line_sum > lines_sum_med:
                mask[i, j] = 255
    blur_mask = cv2.GaussianBlur(mask, (3, 1), 0)
    dilate_mask = cv2.dilate(mask, np.ones((1, 3), np.uint8), iterations=3)

    show_with_resize("mask", mask, 450)
    # show_with_resize("blur_mask", blur_mask, 450)
    show_with_resize("dilate_mask", dilate_mask, 450)
    return dilate_mask


def draw_word_rectangles(img, mask):
    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 75 < w < 500 and 30 < h < 100:
            cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)
    show_with_resize("img contours", img, 450)
    return img


def process(img):
    mask = find_lines_mask(img)
    res = cv2.bitwise_and(img, img, mask=mask)
    show_with_resize("res", res, 450)
    marked_words = draw_word_rectangles(img, mask)
    show_with_resize("marked_words", marked_words, 450)
    cv2.waitKey(0)


def main(path):
    for f_name in os.listdir(path):
        print(f_name)
        if f_name[-3:] == "jpg":
            img = cv2.imread(os.path.join(path, f_name))
            process(img)

        # print(os.path.exists())


if __name__ == "__main__":
    path = os.path.join(".", "data")
    main(path)

