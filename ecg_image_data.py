from pdf2image import convert_from_path
import numpy as np
import cv2
import os
from tqdm import tqdm
import pandas as pd

def pdf2image(pdf_path):
  images = convert_from_path(pdf_path, dpi=600)
  if not images:
    print("No images found in PDF.")

  # Assuming first page is the ECG page
  image = np.array(images[0])
  return image


def preprocess_image(image):
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  # 判断是否为彩色图像
  is_color = not (np.all(image[:, :, 0] == image[:, :, 1]) and
                  np.all(image[:, :, 1] == image[:, :, 2]))
  print(is_color)

  if is_color:
    # 红色背景处理（BGR格式）
    R = image[:, :, 2]
    G = image[:, :, 1]
    B = image[:, :, 0]
    red_mask = (R > 200) & (G < 100) & (B < 100)  # 调整阈值以适应不同情况
    image[red_mask] = [255, 255, 255]  # 替换为白色背景

  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
  cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
  return cleaned,is_color

def lead_segment(x_bias=0,y_bias=0):
  x_start = 669 + x_bias
  x_end = 6581 + x_bias
  d_x = int((x_end - x_start) / 4)
  y_start = 1251 + y_bias
  y_end = 4555 + y_bias
  d_y = int((y_end - y_start) / 4)

  Lead_I=image[y_start + 0 * d_y : y_start + 1 * d_y, x_start + 0* d_x : x_start + 1 * d_x]
  Lead_II=image[y_start + 1 * d_y : y_start + 2 * d_y, x_start + 0* d_x : x_start + 1 * d_x]
  Lead_III=image[y_start + 2 * d_y : y_start + 3 * d_y, x_start + 0* d_x : x_start + 1 * d_x]
  Lead_avR=image[y_start + 0 * d_y : y_start + 1 * d_y, x_start + 1* d_x : x_start + 2 * d_x]
  Lead_avL=image[y_start + 1 * d_y : y_start + 2 * d_y, x_start + 1* d_x : x_start + 2 * d_x]
  Lead_avF=image[y_start + 2 * d_y : y_start + 3 * d_y, x_start + 1* d_x : x_start + 2 * d_x]
  Lead_V1=image[y_start + 0 * d_y : y_start + 1 * d_y, x_start + 2* d_x : x_start + 3 * d_x]
  Lead_V2=image[y_start + 1 * d_y : y_start + 2 * d_y, x_start + 2* d_x : x_start + 3 * d_x]
  Lead_V3=image[y_start + 2 * d_y : y_start + 3 * d_y, x_start + 2* d_x : x_start + 3 * d_x]
  Lead_V4=image[y_start + 0 * d_y : y_start + 1 * d_y, x_start + 3* d_x : x_start + 4 * d_x]
  Lead_V5=image[y_start + 1 * d_y : y_start + 2 * d_y, x_start + 3* d_x : x_start + 4 * d_x]
  Lead_V6=image[y_start + 2 * d_y : y_start + 3 * d_y, x_start + 3* d_x : x_start + 4 * d_x]
  Lead_long_II=image[y_start + 3 * d_y : y_start + 4 * d_y, x_start + 0* d_x : x_start + 4 * d_x]

  return [Lead_I,Lead_II,Lead_III,Lead_avR,Lead_avL,Lead_avF,Lead_V1,Lead_V2,Lead_V3,Lead_V4,Lead_V5,Lead_V6,Lead_long_II]


if __name__ == "__main__":
  # Specify the directory path
  directory = "./data/ecg_data"
  pdf_path = os.path.join(directory, "pdf")
  save_path = os.path.join(directory, "image_datasets/train/")
  os.makedirs(pdf_path, exist_ok=True)
  os.makedirs(save_path, exist_ok=True)

  # List all the files in the directory
  files = os.listdir(pdf_path)

  filenames = []
  labels = []

  for file in tqdm(files):
    image = pdf2image(pdf_path + f"/{file}")
    image_path = os.path.join(save_path,f"{file[:-4]}")
    filename = f"{file[:-4]}"
    filenames.append(filename)
    labels.append("N")
    os.makedirs(image_path,exist_ok=True)
    image,is_color = preprocess_image(image)
    if is_color:
      x_bias = 0  # -55
      y_bias = 0  # -150
    else:
      x_bias = -55
      y_bias = -150


    Leads = lead_segment(x_bias,y_bias)
    lead_grid = {
      "I": 0, "aVR": 3, "V1": 6, "V4": 9,
      "II": 1, "aVL": 4, "V2": 7, "V5": 10,
      "III": 2, "aVF": 5, "V3": 8, "V6": 11,
    }

    for lead in lead_grid.keys():
      cv2.imwrite(os.path.join(image_path, f"{lead}.png"), Leads[lead_grid[lead]])
      print(Leads[lead_grid[lead]].shape)

    metadata = pd.DataFrame({
      "filename": filenames,
      "label": labels
    })
    metadata.to_csv(save_path + "/metadata.csv", index=False)


