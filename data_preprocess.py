from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import glob
import cv2
import numpy as np
from xml.etree import ElementTree as ET

# 0: red, 1: yellow, 2: green
# txt for each image: 
# file_name(=image_name), label, xmin(0~1), xmax(0~1), ymin(0~1), ymax(0~1)

COLOR_LABEL = {
  'Red': '0',
  'Yellow': '1',
  'Green': '2'
}

class Box:
  def __init__(self, x1, y1, x2, y2):
    self.x1 = x1
    self.y1 = y1
    self.x2 = x2
    self.y2 = y2

  def rescale(self, x, y):
    self.x1 *= x
    self.x2 *= x
    self.y1 *= y
    self.y2 *= y

  def shift(self, x, y):
    self.x1 -= x
    self.x2 -= x
    self.y1 -= y
    self.y2 -= y
  
def resize_crop_with_box(image, boxes, imm_h, imm_w, target_h, target_w):
  assert imm_h >= target_h and imm_w >= target_w

  image_h, image_w = image.shape[:2]
  imm_scale_x = imm_w / image_w
  imm_scale_y = imm_h / image_h

  imm_image = cv2.resize(image, (imm_w, imm_h))
  for i, box in enumerate(boxes):
    box.rescale(imm_scale_x, imm_scale_y)
    boxes[i] = box

  crop_h_start = (imm_h - target_h) // 2
  crop_h_end = crop_h_start + target_h
  crop_w_start = (imm_w - target_w) // 2
  crop_w_end = crop_w_start + target_w
  for i, box in enumerate(boxes):
    box.shift(crop_w_start, crop_h_start)
    boxes[i] = box

  result_image = imm_image[crop_h_start:crop_h_end, crop_w_start:crop_w_end, :]
  return result_image, boxes

def resize_images_and_boxes_in_dataset(img_path, output_dir, root, target_w=384, target_h=288):

  image = cv2.imread(img_path)
  image_h, image_w = image.shape[:2]

  outputs = []
  for aug_times in range(20):
    output = []
    boxes = []
    colors = []
    for p in root.findall('.//object'):
      xmin = int(p.find('bndbox/xmin').text)
      ymin = int(p.find('bndbox/ymin').text)
      xmax = int(p.find('bndbox/xmax').text)
      ymax = int(p.find('bndbox/ymax').text)
      boxes.append(Box(xmin, ymin, xmax, ymax))
      color = COLOR_LABEL[p.find('name').text]
      colors.append(color)

    imm_h = target_h + aug_times * 3
    imm_w = int(imm_h * (image_w / image_h))
    img, boxes = resize_crop_with_box(image, boxes, imm_h, imm_w, target_h, target_w)

    basename = os.path.basename(img_path)
    output_path = os.path.join(output_dir, basename[:-4] + '_%d' % aug_times + basename[-4:])
    cv2.imwrite(output_path, img)
    
    output.append(output_path)
    for color, box in zip(colors, boxes):
      output += [color, str(box.x1), str(box.y1), str(box.x2), str(box.y2)]

    outputs.append(output)

  return outputs

def resize_images_and_boxes_in_real_dataset(img_path, output_dir, root, width=384, height=288):
  output = []
  img_width = int(root.find('size/width').text)
  img_height = int(root.find('size/height').text) - 70
  
  img = cv2.imread(img_path)[:-70]
  img = cv2.resize(img, (width, height))
  
  output_path = os.path.join(output_dir, os.path.basename(img_path))
  cv2.imwrite(output_path, img)
  output.append(output_path)

  color_label = {
    'Red': '0',
    'Yellow': '1',
    'Green': '2'
  }

  width_ratio = width / img_width
  height_ratio = height / img_height
  test_dir = os.path.join(output_dir, 'test')
  if not os.path.exists(test_dir):
    os.makedirs(test_dir)
  for p in root.findall('.//object'):
      color = color_label[p.find('name').text]
      xmin = int(p.find('bndbox/xmin').text) / img_width * width
      ymin = int(p.find('bndbox/ymin').text) / img_width * width
      xmax = int(p.find('bndbox/xmax').text) / img_height * height
      ymax = int(p.find('bndbox/ymax').text) / img_height * height
      output.extend([color, str(xmin), str(ymin), str(xmax), str(ymax)])
      cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(255, 0, 0))
      cv2.imwrite(os.path.join(test_dir, os.path.basename(img_path)), img)

  return output


if __name__ == '__main__':
  def run(output_path, image_dir, resize_dir):
    if not os.path.exists(resize_dir):
      os.makedirs(resize_dir)

    if 'simulator' in image_dir:
      with open(output_path, 'w') as f:
          for color in ['Green', 'Red', 'Yellow']:
              dir_path = os.path.join(image_dir, color, 'labels')
              for filepath in glob.glob(dir_path+'/*.xml'):
                  tree = ET.parse(filepath)
                  root = tree.getroot()
                  img_path = os.path.join(image_dir, color, root.findall('filename')[0].text)
                  outputs = resize_images_and_boxes_in_dataset(img_path, resize_dir, root)
                  for output in outputs:
                    f.write(','.join(output) + '\n')

                  break
              break

    elif 'testarea' in image_dir:
      with open(output_path, 'w') as f:
        dir_path = os.path.join(image_dir, 'labels')
        green, red = 0, 0
        for filepath in glob.glob(dir_path+'/*.xml'):
            tree = ET.parse(filepath)
            root = tree.getroot()
            img_path = os.path.join(image_dir, root.findall('filename')[0].text)
            output = resize_images_and_boxes_in_real_dataset(img_path, resize_dir, root)
            f.write(','.join(output) + '\n')
            if output[1] == '0':
              red += 1
            if output[1] == '2':
              green += 1
        print(green, red)
  
  def test(data_path, test_dir):
    with open(data_path, 'r') as f:
      for line in f:
        line = line.strip().split(',')
        image_path = line[0]
        line = line[1:]
        image = cv2.imread(image_path)
        for i in range(len(line) // 5, 5):
          color, x1, y1, x2, y2 = list(map(lambda x: int(round(float(x))), line[i:i+5]))
          if color == 0:
            color = (0, 0, 255)
          elif color == 1:
            color = (0, 255, 255)
          else:
            color = (0, 255, 0)
          cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color)
        
        output_path = os.path.join(test_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, image)

  import sys
  output_path = sys.argv[1] # './udacity-traffic-light-dataset/sim_data.txt'
  image_dir = sys.argv[2] # './udacity-traffic-light-dataset/simulator_dataset_rgb/'
  resize_dir = sys.argv[3] # './udacity-traffic-light-dataset/simulator_resize/'
  test_dir = sys.argv[4] # './udacity-traffic-light-dataset/test'

  for directory in [resize_dir, test_dir]:
    if not os.path.exists(directory):
      os.makedirs(directory)

  run(output_path, image_dir, resize_dir)
  test(output_path, test_dir)
