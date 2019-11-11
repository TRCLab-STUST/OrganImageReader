import OrganImageReader as oir
import cv2
import os
import glob

# 使否顯示除錯訊息
debug = True


def main():
    # 各種路徑
    ROOT_DIR = os.path.abspath("../")
    IMAGES_DIR = os.path.join(ROOT_DIR, "resource/(VKH) Segmented Images (1,000 X 570)/")
    JSON_PATH = os.path.join(ROOT_DIR, "json/output.json")
    TABLE_PATH = os.path.join(ROOT_DIR, "resource/color.txt")

    # 開json檔
    json = open(JSON_PATH, 'w')
    json.write("{ \n")

    organ_reader = oir.OrganImageReader(debug)
    # 讀取資料表
    organ_reader.load_table(TABLE_PATH)
    # 讀取整個資料夾的.bmp
    images = glob.glob(IMAGES_DIR + "*.bmp")
    # 逐張
    for image in images:
        # 讀取圖片
        organ_reader.load_image(image)
        # 顯示原始圖片
        cv2.imshow("Origin Image", organ_reader.image_origin)
        # 找出圖片的器官
        organ_reader.find_organ()
        # 建立過濾後器官圖片
        organ_reader.filter_organ(297)
        # 顯示過濾後圖片
        cv2.imshow("Filter Image", organ_reader.image_filter)
        # 建立過濾後圖片輪廓
        image = organ_reader.draw_contours()
        # 顯示繪製完輪廓的圖片
        if image is not None:
            cv2.imshow('Contours', image)


if __name__ == '__main__':
    main()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
