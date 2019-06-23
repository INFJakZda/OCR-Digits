import glob
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import os
import pandas as pd

from scipy.misc import imshow, imsave, imread, imresize
from scipy.ndimage import filters, morphology
from skimage import filters
from skimage.draw import line
from scipy import ndimage, spatial

# load json and create model
json_file = open('model_digit_rec.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_digit_rec.h5")
print("Loaded model from disk")

# X_test = pd.read_csv('../data/test_train.csv')
# X_test.describe()
# X_test = X_test.values.reshape(X_test.shape[0],28,28,1)
# y_pred = loaded_model.predict_classes(X_test, verbose=1)
# print(y_pred)


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

    grad = apply_sobel(gray)

    blur_grad = cv2.medianBlur(grad, ksize_median_blur)
    kernel = np.ones(kernel_size_dilate, np.uint8)

    # erode = cv2.erode(blur_grad, kernel, iterations=3)
    dilate = cv2.dilate(blur_grad, kernel, iterations=5)

    mask = np.zeros(dilate.shape, dtype=np.uint8)
    lines_sum = np.sum(dilate, axis=1)
    lines_sum_mean = np.mean(lines_sum)
    lines_sum_med = np.median(lines_sum)
    for i, line_sum in enumerate(lines_sum):
        line = dilate[i]
        line_mean = np.mean(line)
        line_med = np.median(line)
        for j, pixel in enumerate(line):
            if pixel > line_med * 1.6 and line_sum > lines_sum_med:
                mask[i, j] = 255
    blur_mask = cv2.GaussianBlur(mask, (3, 1), 0)
    dilate_mask = cv2.dilate(mask, np.ones((1, 3), np.uint8), iterations=3)

    return dilate_mask


def draw_word_rectangles(img, mask):
    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 75 < w < 500 and 30 < h < 100:
            cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)
    show_with_resize("img contours", img, 450)
    return img

def detect_words_on_mask(img, mask):
    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    returned_mask = np.zeros_like(mask)
    indexing_height = 0
    index = 0
    tolerance = 20  # TODO
    for contour in reversed(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if 75 < w < 500 and 30 < h < 100:
            diference = (abs(indexing_height - (y + h / 2)))
            if diference > tolerance:
                index += 1    
            indexing_height = (y + h / 2)
            min_y = y
            max_y = (y + h)
            min_x = x 
            max_x = (x + w)
            returned_mask[min_y:max_y,min_x:max_x] = index * 10
    # show_with_resize("img", img, 450)
    # show_with_resize("returned mask", returned_mask, 450)
    # cv2.waitKey(0)
    return returned_mask

def show_with_resize_28(winname, img):
    img = cv2.resize(img, (28,28), interpolation=cv2.INTER_CUBIC)
    cv2.imshow(winname, img)

def houghLines(img):
    w,h = img.shape
    acc=[]
    for i in range(h):
        rr,cc = line(0, i, w-1, h-i-1)
        acc.append(np.sum(img[rr, cc]))
    mi = np.argmax(acc)
    ret = np.zeros(img.shape, dtype=np.bool)
    rr,cc = line(0, mi, w-1, h-mi-1)
    ret[rr,cc]=True
    return ret

def removeLines(img):
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    imfft = np.fft.fft2(imggray)
    imffts = np.fft.fftshift(imfft)

    mags = np.abs(imffts)
    angles = np.angle(imffts)

    visual = np.log(mags)


    visual3 = np.abs(visual.astype(np.int16) - np.mean(visual))

    ret = houghLines(visual3)
    ret = morphology.binary_dilation(ret )
    ret = morphology.binary_dilation(ret )
    ret = morphology.binary_dilation(ret )
    ret = morphology.binary_dilation(ret )
    ret = morphology.binary_dilation(ret )
    w,h=ret.shape
    ret[int(w/2-3):int(w/2+3), int(h/2-3):int(h/2+3)]=False


    delta = np.mean(visual[ret]) - np.mean(visual)


    visual_blured = ndimage.gaussian_filter(visual, sigma=5)



    visual[ret] =visual_blured[ret]


    newmagsshift = np.exp(visual)

    newffts = newmagsshift * np.exp(1j*angles)

    newfft = np.fft.ifftshift(newffts)

    imrev = np.fft.ifft2(newfft)

    newim2 =  np.abs(imrev).astype(np.uint8)


    # newim2 = np.maximum(newim2, img)

    return newim2, img

def to_grey(img, name):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                cv2.THRESH_BINARY, 15, -2)
    # Show binary image
    cv2.imshow("binary" + name, bw)

def get_digits_from_index(img, mask, name):
    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    # sorted_contour = sorted(contour_sizes, key=lambda tup: tup[1])
    sorted_contour = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    if len(sorted_contour) > 6:
        sorted_contour = sorted_contour[:6]

    boundingBoxes = [cv2.boundingRect(c) for c in sorted_contour]
    (sorted_contour, boundingBoxes) = zip(*sorted(zip(sorted_contour, boundingBoxes),
		key=lambda b:b[1][0], reverse=False))

    digits = []
    for idx, box in enumerate(boundingBoxes):
        x, y, w, h = box
        # cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)
        min_y = y
        max_y = (y + h)
        min_x = x
        max_x = (x + w)
        
        digit = mask[min_y:max_y,min_x:max_x]
        # cv2.imshow(str(idx), digit)
        digit = cv2.resize(digit, dsize=(28, 28))
        digit = digit.reshape([1, 28, 28, 1])
        prediction = loaded_model.predict_classes(digit)
        digits.append(prediction[0])
    print(digits)
    cv2.imshow("img contours_final", img)
    
    return ''.join(str(e) for e in digits)

def get_index(gray, name):
    gray = cv2.bitwise_not(gray)
    _, bw = cv2.threshold(gray,135,255,cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    cv2.imshow("binary" + name, bw)
    return get_digits_from_index(gray, bw, "1")


def detect_index(img):
    cv2.imshow("wycinek", img)
    to_grey(img, "wyc")
    
    img2, img3 = removeLines(img)

    cv2.imshow("wycinek2", img2)
    index = get_index(img2, "wyc2")

    cv2.waitKey(0)
    return index

def detect_indexes(img, mask):
    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    indexing_height = 0
    index = 0
    go = False
    min_x = 0
    tolerance = 20  # TODO

    results_recogn_rows = []

    for contour in reversed(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if 75 < w < 500 and 30 < h < 100:
            diference = (abs(indexing_height - (y + h / 2)))
            if diference > tolerance:
                index += 1
                if(index > 1):
                    index_rec = detect_index(img[min_y:max_y,min_x:max_x])
                    results_recogn_rows.append(('', '', index_rec))
                    go = True
            if (x > min_x or go):
                go = False
                min_y = y
                max_y = (y + h)
                min_x = x 
                max_x = (x + w)
            indexing_height = (y + h / 2)
    return results_recogn_rows

def find_paper(large, margin_left=20, margin_right=20, margin_top=20, margin_down=20):
    rgb = cv2.pyrDown(large)
    small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    _, contours, _ = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    mask = np.zeros(bw.shape, dtype=np.uint8)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    x, y, w, h = cv2.boundingRect(biggest_contour)
    new_large = np.zeros_like(large)
    mask[y:y+h, x:x+w] = 0
    if w > 500 and h > 700:
        cv2.rectangle(rgb, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)
        min_y = y * 2 + margin_top
        max_y = (y + h) * 2 - margin_down
        min_x = x * 2 + margin_left
        max_x = (x + w) * 2 - margin_right
        paper_frag = large[min_y:max_y,min_x:max_x]
    else:
        times = 2
        y, x, _ = large.shape
        min_y = margin_top * times
        max_y = y - margin_down * times
        min_x = margin_left * times
        max_x = x - margin_right * times
        paper_frag = large[min_y:max_y,min_x:max_x]
    new_large[min_y:max_y,min_x:max_x] = paper_frag
    return new_large

def detect_words(f_name):
    img = cv2.imread(f_name)

    img_paper = find_paper(img)
    
    mask = find_lines_mask(img_paper)
    _ = cv2.bitwise_and(img_paper, img_paper, mask=mask)
    
    marked_words = detect_words_on_mask(img_paper, mask)

    return marked_words

def ocr(f_name):
    img = cv2.imread(f_name)

    img_paper = find_paper(img)
    
    mask = find_lines_mask(img_paper)
    _ = cv2.bitwise_and(img_paper, img_paper, mask=mask)
    
    results_recogn_rows = detect_indexes(img_paper, mask)

    return results_recogn_rows
