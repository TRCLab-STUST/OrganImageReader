import OrganImageReader as oir
import cv2
import os
import glob
import json

# 使否顯示除錯訊息
debug = False


def main():
    # 各種路徑
    ROOT_DIR = os.path.abspath("../")
    IMAGES_DIR = os.path.join(ROOT_DIR, "resource/(VKH) Segmented Images (1,000 X 570)/")
    JSON_PATH = os.path.join(ROOT_DIR, "json/output.json")
    TABLE_PATH = os.path.join(ROOT_DIR, "resource/color.txt")
    FIND_INDEX = 784

    organ_reader = oir.OrganImageReader(debug)
    # 讀取資料表
    organ_reader.load_table(TABLE_PATH)
    # 讀取整個資料夾的.bmp
    images = glob.glob(IMAGES_DIR + "*.bmp", recursive=True)
    # 逐張
    for image in images:
        # 取得檔案名子
        filename = os.path.basename(image)
        # 讀取圖片
        organ_reader.load_image(image)
        # 顯示原始圖片
        # cv2.imshow("Origin Image", organ_reader.image_origin)
        # 找出圖片的器官
        organ_reader.find_organ()
        # 建立過濾後器官圖片
        organ_reader.filter_organ(FIND_INDEX)
        # 顯示過濾後圖片
        # cv2.imshow("Filter Image", organ_reader.image_filter)
        # 建立過濾後圖片輪廓
        image = organ_reader.draw_contours()
        # 顯示繪製完輪廓的圖片
        # if image is not None:
        #     cv2.imshow('Contours', image)

        json_file = open(JSON_PATH, 'r')
        json_file_data = json_file.read()

        data = json.loads(json_file_data)
        key = filename + str(organ_reader.image_size)
        data[key] = {}
        data[key]['fileref'] = ''
        data[key]['size'] = organ_reader.image_size
        data[key]['filename'] = filename
        data[key]['base64_img_data'] = ''
        data[key]['file_attributes'] = {}
        data[key]['regions'] = {}

        for n in range(0, len(organ_reader.contours)):
            list_x = []
            list_y = []
            for point in organ_reader.contours[n]:
                for x, y in point:
                    list_x.append(x)
                    list_y.append(y)

            data[key]['regions'][n] = {}
            data[key]['regions'][n]['shape_attributes'] = {}
            data[key]['regions'][n]['shape_attributes']['name'] = 'polygon'
            data[key]['regions'][n]['shape_attributes']['all_points_x'] = list_x
            data[key]['regions'][n]['shape_attributes']['all_points_y'] = list_y
            data[key]['regions'][n]['region_attributes'] = {}
            data[key]['regions'][n]['region_attributes']['name'] = FIND_INDEX

        with open(JSON_PATH, "w") as file_write:
            json.dump(data, file_write, default=int)


if __name__ == '__main__':
    main()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
