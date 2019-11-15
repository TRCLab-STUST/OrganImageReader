import OrganImageReader as oir
import cv2
import os
import numpy as np

debug = False

ROOT_DIR = os.path.abspath("../")
TABLE_PATH = os.path.join(ROOT_DIR, "resource/color.txt")
vkh = "../resource/(VKH) Segmented Images (1,000 X 570)/0125.bmp"
ct = "../resource/CT/0125.jpg"
vkhimg = cv2.imread(vkh)
ctimg = cv2.imread(ct)

vh, vw = np.shape(vkhimg)[:2]
ch, cw = np.shape(ctimg)[:2]
scaleh, scalew = [round(vh / ch), round(vw / cw)]
print("Vkh = " + str(vh) + "X" + str(vw))
print("Ct = " + str(ch) + "X" + str(cw))
print("scale = " + str(scaleh) + "X" + str(scalew))




organ_reader = oir.OrganImageReader(debug)
# 讀取資料表
organ_reader.load_table(TABLE_PATH)

organ_reader.load_image(vkh)
# 找出圖片的器官
organ_reader.find_organ()

organ_reader.filter_organ(2)

image_contours = organ_reader.draw_contours()

if image_contours is not None:
    cv2.drawContours(vkhimg, organ_reader.contours, -1, (0, 0, 255), 1)
    cv2.imshow('vkh', vkhimg)

    for contour in organ_reader.contours:
        contour[:, :, 0] = contour[:, :, 0] * 0.5
        contour[:, :, 1] = contour[:, :, 1] * 0.5

    cv2.drawContours(ctimg, contour, -1, (0, 0, 255), 1)
    cv2.imshow('ct', ctimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
