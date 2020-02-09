import os
import glob
import cv2
from xml.etree import ElementTree as ET

# 0: red, 1: yellow, 2: green
# txt for each image: 
# file_name(=image_name), label, xmin(0~1), xmax(0~1), ymin(0~1), ymax(0~1)

def resize_images_and_boxes_in_dataset(img_path, output_dir, root, width=384, height=288):
  output = []
  img_width = int(root.find('size/width').text)
  img_height = int(root.find('size/height').text)
  
  img = cv2.imread(img_path)
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
  def run(image_dir, resize_dir):
    if not os.path.exists(resize_dir):
      os.makedirs(resize_dir)

    with open('./udacity-traffic-light-dataset/train.txt', 'w+') as f:
        for color in ['Green', 'Red', 'Yellow']:
            dir_path = os.path.join(image_dir, color, 'labels')
            for filepath in glob.glob(dir_path+'/*.xml'):
                tree = ET.parse(filepath)
                root = tree.getroot()
                img_path = os.path.join(image_dir, color, root.findall('filename')[0].text)
                output = resize_images_and_boxes_in_dataset(img_path, resize_dir, root)
                f.write(','.join(output) + '\n')

  import sys
  image_dir = sys.argv[1] # './udacity-traffic-light-dataset/simulator_dataset_rgb/'
  resize_dir = sys.argv[2] # './udacity-traffic-light-dataset/simulator_resize/'
  run(image_dir, resize_dir)