import cv2
import os
def add_magnifier_corner(image_path, x, y, w, h, zoom):
    # x, y ROI 左上角坐标
    # w, h ROI 宽和高
    #zoom # 放大倍数

    # 读图
    img = cv2.imread(image_path)
    assert img is not None, '图片路径不对'

    # 截取并放大
    roi = img[y:y+h, x:x+w]
    roi_big = cv2.resize(roi, (w*zoom, h*zoom), interpolation=cv2.INTER_CUBIC)

    # 贴到右下角
    H, W = img.shape[:2]
    bx, by = W - w*zoom - 10, H - h*zoom - 10   # 10 像素留边
    img[by:by+h*zoom, bx:bx+w*zoom] = roi_big

    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 5)
    cv2.rectangle(img, (bx, by), (bx+w*zoom, by+h*zoom), (0, 255, 0), 5)

    cv2.imwrite(image_path[:-4]+'_1.png', img)
    cv2.waitKey(0)

if __name__ == "__main__":
    image_path = "Input_RAW.png"
    add_magnifier_corner(image_path=image_path, x=720, y=630, w=280,h=250,zoom=3)