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
    # show_with_resize("normal", gray, 450)

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
    # print("median: {}, mean: {}".format(lines_sum_mean, lines_sum_med))
    for i, line_sum in enumerate(lines_sum):
        line = dilate[i]
        line_mean = np.mean(line)
        line_med = np.median(line)
        for j, pixel in enumerate(line):
            if pixel > line_med * 1.6 and line_sum > lines_sum_med:
                mask[i, j] = 255
    blur_mask = cv2.GaussianBlur(mask, (3, 1), 0)
    dilate_mask = cv2.dilate(mask, np.ones((1, 3), np.uint8), iterations=3)

    # show_with_resize("mask", mask, 450)
    # show_with_resize("blur_mask", blur_mask, 450)
    # show_with_resize("dilate_mask", dilate_mask, 450)
    return dilate_mask


def draw_word_rectangles(img, mask):
    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 75 < w < 500 and 30 < h < 100:
            cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)
    # show_with_resize("img contours", img, 450)
    return img

def detect_words_on_mask(img, mask):
    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    returned_mask = np.zeros_like(mask)
    indexing_height = 0
    index = 0
    tolerance = 10  # TODO
    for contour in reversed(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if 75 < w < 500 and 30 < h < 100:
            diference = (abs(indexing_height - (y + h / 2)))
            print(diference)
            if diference > tolerance:
                index += 1    
            indexing_height = (y + h / 2)
            min_y = y
            max_y = (y + h)
            min_x = x 
            max_x = (x + w)
            # show_with_resize("large", large, 450)
            returned_mask[min_y:max_y,min_x:max_x] = index * 10
    # show_with_resize("img contours", img, 450)
    print("INDEX")
    print(index)
    # show_with_resize("img", img, 450)
    # show_with_resize("returned mask", returned_mask, 450)
    # cv2.waitKey(0)
    return returned_mask

def find_paper(large, margin_left=20, margin_right=20, margin_top=20, margin_down=20):
    rgb = cv2.pyrDown(large)
    small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("asf", connected)
    # cv2.waitKey(0)
    # using RETR_EXTERNAL instead of RETR_CCOMP
    _, contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    mask = np.zeros(bw.shape, dtype=np.uint8)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    x, y, w, h = cv2.boundingRect(biggest_contour)
    new_large = np.zeros_like(large)
    mask[y:y+h, x:x+w] = 0
    if w > 500 and h > 700:
        cv2.rectangle(rgb, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)
        show_with_resize("rect", rgb, 450)
        min_y = y * 2 + margin_top
        max_y = (y + h) * 2 - margin_down
        min_x = x * 2 + margin_left
        max_x = (x + w) * 2 - margin_right
        paper_frag = large[min_y:max_y,min_x:max_x]
        # show_with_resize("rgb", rgb, 450)
        # show_with_resize("large", large, 450)
        
    else:
        times = 2
        y, x, _ = large.shape
        min_y = margin_top * times
        max_y = y - margin_down * times
        min_x = margin_left * times
        max_x = x - margin_right * times
        paper_frag = large[min_y:max_y,min_x:max_x]
    new_large[min_y:max_y,min_x:max_x] = paper_frag
    show_with_resize("new_large", new_large, 450)
    # cv2.imshow('rects', large)
    # print(str(w) + " " + str(h))
    # cv2.imwrite(batch + "-results/" + f_name, large)
    return new_large

def detect_words(f_name):
    img = cv2.imread(f_name)
    img_paper = find_paper(img)
    # show_with_resize("img_paper", img_paper, 450)
    mask = find_lines_mask(img_paper)
    res = cv2.bitwise_and(img_paper, img_paper, mask=mask)
    # show_with_resize("res", res, 450)
    marked_rects = draw_word_rectangles(img_paper, mask)
    show_with_resize("marked_rects", marked_rects, 450)
    marked_words = detect_words_on_mask(img_paper, mask)
    show_with_resize("marked_words", marked_words, 450)
    # show_with_resize("marked_words", marked_words, 450)
    cv2.waitKey(0)
    return marked_words
