# -*- coding: utf-8 -*-
# импортируйте необходимые пакеты
import numpy as np
import cv2
import requests as req
import imutils
from fann2 import libfann as fann
import os

ENG_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
MODULE_PATH = os.path.dirname(os.path.realpath(__file__)) + '/'

ann = fann.neural_net()
ann.create_from_file(MODULE_PATH + 'captcha.net')


def solve(url):
    open(MODULE_PATH + 'urls.txt', 'a').write(url + '\n')
    image = req.get(url).content
    image = np.asarray(bytearray(image), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Add some extra padding around the image
    gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

    # threshold the image (convert it to pure black and white)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # find the contours (continuous blobs of pixels) the image
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Hack for compatibility with different OpenCV versions
    contours = contours[0] if imutils.is_cv2() else contours[1]

    letter_image_regions = []

    # Now we can loop through each of the four contours and extract the letter
    # inside of each one
    for contour in contours:
        # Get the rectangle that contains the contour
        (x, y, w, h) = cv2.boundingRect(contour)

        # Compare the width and height of the contour to detect letters that
        # are conjoined into one chunk
        if w / h > 1.25:
            # This contour is too wide to be a single letter!
            # Split it in half into two letter regions!
            half_width = int(w / 2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))
        else:
            # This is a normal letter by itself
            letter_image_regions.append((x, y, w, h))

    letter_image_regions.sort(key=lambda x: x[0], reverse=False)

    text = ''
    for letter_bounding_box in letter_image_regions:
        print(letter_bounding_box)
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box

        # Extract the letter from the original image with a 2-pixel margin around the edge
        x1, y1 = x - 2, y - 2
        if y < 2:
            y1 = y + 2
        if x < 2:
            x1 = x + 2
        letter_image = gray[y1:y + h + 2, x1:x + w + 2]

        letter_image = cv2.resize(letter_image, (35, 35))
        letter_image = cv2.threshold(letter_image, 0, 255, cv2.THRESH_BINARY)[1]
        img_bin = []
        for x in letter_image:
            for y in x:
                if y == 0:
                    img_bin.append(1)
                else:
                    img_bin.append(0)

        res = ann.run(img_bin)
        text += ENG_ALPHABET[res.index(max(res))]
        # cv2.imwrite('./'+ENG_ALPHABET[res.index(max(res))]+'.jpg', letter_image)

    return text


if __name__ == '__main__':
    for url in open('./urls.txt').readlines():
        print(url[:-1])
        res = solve(url)
        print(res + '\n')
