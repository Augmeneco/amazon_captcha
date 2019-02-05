from fann2 import libfann as fann
import os
import cv2 as cv

ann = fann.neural_net()
ann.create_from_file('./captcha.net')

ENG_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
glob_counter = 0
glob_letters_len = 0
letters_percents = {}
for letter in os.listdir('./letters/'):
    print('----\n' + letter)
    counter = 0
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

        # for i, val in enumerate(img_bin):
        #     print(val, end='')
        #     if (i % 35) == 0:
        #         print('')

        res = ann.run(img_bin)
        if ENG_ALPHABET[res.index(max(res))] == letter:
            counter += 1
            glob_counter += 1
        print(letter + ' = ' + ENG_ALPHABET[res.index(max(res))])
    letters_len = len(os.listdir('./letters/' + letter))
    glob_letters_len += letters_len

    if letters_len != 0:
        letters_percents[letter] = (100. / letters_len) * counter
        print("%.2f" % letters_percents[letter] + '%')
    else:
        letters_percents[letter] = -1
        print('-1%')
print('-----\n')
print("%.2f" % ((100. / glob_letters_len) * glob_counter) + '%')
letters_percents = sorted(letters_percents, key=lambda x: x.value(), reverse=True)
print(letters_percents)
