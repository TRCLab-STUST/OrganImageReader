import OrganImageReader as oir
import cv2
import os
import numpy as np

debug = True

FIND_INDEX = np.arange(40, 64)
ROOT_DIR = os.path.abspath("../")
TABLE_PATH = os.path.join(ROOT_DIR, "resource/color.txt")
vkh = "../resource/(VKH) Segmented Images (1,000 X 570)/2795.bmp"
ct = "../resource/CT/0125.jpg"
bigger = "../resource/BiggerCT/0125.jpg"
vkhimg = cv2.imread(vkh)
# ctimg = cv2.imread(ct)
# biggerimg = cv2.imread(bigger)

# vh, vw = np.shape(vkhimg)[:2]
# ch, cw = np.shape(ctimg)[:2]
# bh, bw = np.shape(biggerimg)[:2]
# scaleh, scalew = [round(vh / ch), round(vw / cw)]

# print("Vkh = " + str(vh) + "X" + str(vw))
# print("Ct = " + str(ch) + "X" + str(cw))
# print("Bigger = " + str(bh) + "X" + str(bw))
# print("scale = " + str(scaleh) + "X" + str(scalew))


organ_reader = oir.OrganImageReader(debug)
# 讀取資料表
organ_reader.load_table(TABLE_PATH)

organ_reader.load_image(vkh)
# 找出圖片的器官
organ_reader.find_organ()

for idx in FIND_INDEX:
    organ_reader.filter_organ(idx)
    image_contours = organ_reader.draw_contours()
    cv2.drawContours(vkhimg, organ_reader.contours, -1, (0, 0, 255), 1)

cv2.imshow('farm', vkhimg)
cv2.waitKey(0)