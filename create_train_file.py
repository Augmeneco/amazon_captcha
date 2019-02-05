import os
from fann2 import libfann as fann
import cv2 as cv

ENG_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
outputs = []
inputs = []
for letter in os.listdir('./letters/'):
    print(letter)
    for letter_img in os.listdir('./letters/' + letter):
        img = cv.imread(os.path.join('./letters', letter, letter_img))
        img = cv.resize(img, (35, 35))
        img = cv.threshold(img, 0, 255, cv.THRESH_BINARY)[1]
        img_bin = []
        for x in img:
            for y in x:
                if sum(y) == 0:
                    img_bin.append(1)
                else:
                    img_bin.append(0)
        result_array = ([0] * 26)
        result_array[ENG_ALPHABET.index(letter)] = 1
        outputs.append(result_array)
        inputs.append(img_bin)

td = fann.training_data()
td.set_train_data(inputs, outputs)
td.save_train('./train.dat')
