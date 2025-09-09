import cv2
import os
from PIL import Image, ImageDraw
import numpy as np
import random

def add_magnifier_to_one_image(src_path, dst_path, x, y, w, h, zoom):
    """单图处理：src_path -> dst_path"""
    img = cv2.imread(src_path)
    if img is None:
        print(f'[警告] 无法读取：{src_path}')
        return False

    # 截取并放大
    roi = img[y:y + h, x:x + w]
    roi_big = cv2.resize(roi, (w * zoom, h * zoom), interpolation=cv2.INTER_CUBIC)

    # 右下角位置
    H, W = img.shape[:2]
    bx = max(W - w * zoom - 10, 0)
    by = max(H - h * zoom - 10, 0)
    roi_big = roi_big[:H - by, :W - bx]  # 防越界

    # 贴图
    img[by:by + roi_big.shape[0], bx:bx + roi_big.shape[1]] = roi_big

    # 画框
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 5)
    cv2.rectangle(img, (bx, by),
                  (bx + roi_big.shape[1], by + roi_big.shape[0]),
                  (0, 255, 0), 5)

    # 保存
    cv2.imwrite(dst_path, img)
    print(f'已保存：{dst_path}')
    return True

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

def add_magnifier_to_folder(folder_path, x, y, w, h, zoom):
    """
    批量处理：folder_path 下所有图片 -> folder_path/mag/*.png
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f'路径不存在：{folder_path}')

    mag_dir = os.path.join(folder_path, 'mag')
    os.makedirs(mag_dir, exist_ok=True)

    for name in os.listdir(folder_path):
        ext = os.path.splitext(name)[1].lower()
        if ext not in IMG_EXTS:
            continue
        src = os.path.join(folder_path, name)
        dst_name = os.path.splitext(name)[0] + '.png'   # 统一扩展名
        dst = os.path.join(mag_dir, dst_name)
        add_magnifier_to_one_image(src, dst, x, y, w, h, zoom)

def add_border_to_cropped_image(cropped_img, border_width, border_color):
    """
    给裁剪后的图片添加边框。

    :param cropped_img: 裁剪后的图片对象
    :param border_width: 边框的宽度
    :param border_color: 边框的颜色
    :return: 添加边框后的图片对象
    """
    # 创建一个新的图片对象，宽度和高度比原裁剪图片大边框的两倍
    new_width = cropped_img.width + border_width * 2
    new_height = cropped_img.height + border_width * 2
    new_img = Image.new('RGB', (new_width, new_height), border_color)
    
    # 计算裁剪图片在新图片中的位置
    position = (border_width, border_width)
    
    # 将裁剪图片粘贴到新图片上
    new_img.paste(cropped_img, position)
    
    return new_img

def crop_highlight_and_border_images(folder_path, output_folder, cropped_output_folder, crop_size=(100, 100), crop_position=(50, 50), line_width=5, color='red', border_width=10, border_color='black'):
    """
    对指定文件夹中的所有图片进行操作：
    1. 裁剪选定区域并保存为单独的图片，周围添加边框。
    2. 在原图上绘制边框并保存。

    :param folder_path: 包含原始图片的文件夹路径
    :param output_folder: 保存标记后的图片的文件夹路径
    :param cropped_output_folder: 保存裁剪区域的文件夹路径
    :param crop_size: 裁剪区域的大小，格式为(width, height)
    :param crop_position: 裁剪区域的起始位置，格式为(x, y)
    :param line_width: 边框的线宽
    :param color: 边框的颜色
    :param border_width: 裁剪图片边框的宽度
    :param border_color: 裁剪图片边框的颜色
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(cropped_output_folder):
        os.makedirs(cropped_output_folder)

    # 遍历文件夹中的所有图片
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            with Image.open(image_path) as img:
                # 裁剪图片的指定区域
                cropped_img = img.crop((crop_position[0], crop_position[1],
                                        crop_position[0] + crop_size[0],
                                        crop_position[1] + crop_size[1]))
                
                # 给裁剪后的图片添加边框
                bordered_img = add_border_to_cropped_image(cropped_img, border_width, border_color)
                
                # 保存带边框的裁剪图片
                cropped_filename = os.path.splitext(filename)[0] + '_cropped_bordered' + os.path.splitext(filename)[1]
                bordered_img.save(os.path.join(cropped_output_folder, cropped_filename))
                print(f"Saved bordered cropped image: {os.path.join(cropped_output_folder, cropped_filename)}")

                # 创建一个ImageDraw对象
                draw = ImageDraw.Draw(img)
                # 计算边框的坐标
                x1, y1 = crop_position
                x2, y2 = (x1 + crop_size[0], y1 + crop_size[1])
                # 绘制矩形边框
                draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
                # 保存标记后的图片
                highlighted_filename = os.path.splitext(filename)[0] + '_highlighted' + os.path.splitext(filename)[1]
                img.save(os.path.join(output_folder, highlighted_filename))
                print(f"Saved highlighted image: {os.path.join(output_folder, highlighted_filename)}")



if __name__ == "__main__":
    
    add_magnifier_to_folder(folder_path="E:\Deep Learning\PGRawFormer\\20008_00_0.04",
                            x=800, y=400, w=400, h=400, zoom=3)
    #folder_path = 'E:\Deep Learning\HiMA\RAW data\\visual_results\\10140_00_0.1'
    #output_folder = 'E:\Deep Learning\HiMA\RAW data\\visual_results\\10140_00_0.1\hightlighted'
    #cropped_output_folder = 'E:\Deep Learning\HiMA\RAW data\\visual_results\\10140_00_0.1\hightlighted'
    #crop_highlight_and_border_images(folder_path, output_folder, cropped_output_folder,
    # crop_size=(400, 400), crop_position=(2150, 300),
    # line_width=5, color='blue', border_width=10, border_color='blue')
