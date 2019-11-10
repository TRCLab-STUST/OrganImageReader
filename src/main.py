import OrganImageReader as oir
import cv2

# 使否顯示除錯訊息
debug = True


def main():
    organ_reader = oir.OrganImageReader(debug)
    # 讀取資料表
    organ_reader.load_table('../resource/color.txt')
    # 讀取圖片
    organ_reader.load_image('../resource/BODY/(VKH) Segmented Images (1,000 X 570)/1570.bmp')
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
