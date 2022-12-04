import cv2
from glob import glob
import os

png_folder_path = '변환할 png 이미지가 저장된 경로'

jpg_folder_path = 'jpg 이미지가 저장될 경로'

png_file_list = glob(png_folder_path+'/*.png')

for k in png_file_list:
    png_img = cv2.imread(k)
    png_file_name = k.split('/')[-1]
    new_file_name = png_file_name.replace('png','jpg')
    jpg_file_path = os.path.join(jpg_folder_path,new_file_name)
    cv2.imwrite(jpg_file_path,png_img)