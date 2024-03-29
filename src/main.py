import OrganImageReader as oir
import cv2
import os
import glob
import json
import numpy as np

# 使否顯示除錯訊息
debug = False


def main():
    # 各種路徑
    ROOT_DIR = os.path.abspath("../")
    IMAGES_DIR = os.path.join(ROOT_DIR, "resource/")
    COLOR_DIR = os.path.join(IMAGES_DIR, "color/")
    ORG_DIR = os.path.join(IMAGES_DIR, "org/")
    JSON_PATH = os.path.join(ROOT_DIR, "json/output.json")
    TABLE_PATH = os.path.join(ROOT_DIR, "resource/color.txt")
    FIND_INDEX = np.arange(40, 68)

    organ_reader = oir.OrganImageReader(debug)
    # 讀取資料表
    organ_reader.load_table(TABLE_PATH)
    # 讀取整個資料夾的.bmp
    images = glob.glob(COLOR_DIR + "*.bmp", recursive=True)

    # 創建.Json
    json_open = open(JSON_PATH, 'w')
    json_open.write("{\n}")
    json_open.close()
    json_file = open(JSON_PATH, 'r')
    json_file_data = json_file.read()
    data = json.loads(json_file_data)

    # 逐張
    for image in images:
        # 取得檔案名字
        filename = os.path.basename(image)
        filename = filename[:-3]
        filename = filename + "jpg"
        print("Image: " + filename + "\n")
        # 讀取圖片
        organ_reader.load_image(image)
        # 找出圖片的器官
        organ_reader.find_organ()
        size = os.path.getsize(ORG_DIR + filename)
        key = filename + str(size)
        data[key] = {}
        data[key]['fileref'] = ''
        data[key]['size'] = size
        data[key]['filename'] = filename
        data[key]['base64_img_data'] = ''
        data[key]['file_attributes'] = {}
        data[key]['regions'] = {}
        a = 0
        for idx in FIND_INDEX:
            # 建立過濾後器官圖片
            organ_reader.filter_organ(idx)
            # 建立過濾後圖片輪廓
            image_contours = organ_reader.draw_contours()

            for n in range(0, len(organ_reader.contours)):
                list_x = []
                list_y = []
                for point in organ_reader.contours[n]:
                    for x, y in point:
                        list_x.append(x)
                        list_y.append(y)

                data[key]['regions'][a] = {}
                data[key]['regions'][a]['shape_attributes'] = {}
                data[key]['regions'][a]['shape_attributes']['name'] = 'polygon'
                data[key]['regions'][a]['shape_attributes']['all_points_x'] = list_x
                data[key]['regions'][a]['shape_attributes']['all_points_y'] = list_y
                data[key]['regions'][a]['region_attributes'] = {}
                data[key]['regions'][a]['region_attributes']['name'] = str(idx)
                a += 1

    with open(JSON_PATH, "w") as file_write:
        json.dump(data, file_write, default=int)


if __name__ == '__main__':
    main()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
