import os
import glob
from xml.etree import ElementTree as ET

# 0: red, 1: yellow, 2: green
# txt for each image: 
# file_name(=image_name), label, xmin(0~1), xmax(0~1), ymin(0~1), ymax(0~1)

dir = './udacity-traffic-light-dataset/simulator_dataset_rgb/'

with open('./udacity-traffic-light-dataset/train.txt', 'w+') as f:
    for color in ['Green', 'Red', 'Yellow']:
        dir_path = os.path.join(dir, color, 'labels')
        for filepath in glob.glob(dir_path+'/*.xml'):
            tree = ET.parse(filepath)
            root = tree.getroot()
            output = []
            output.append(os.path.join(dir, color, root.findall('filename')[0].text))
            output.append(root.find('size/width').text)
            output.append(root.find('size/height').text)
            for p in root.findall('.//object'):
                output.append(p.find('name').text)
                output.append(p.find('bndbox/xmin').text)
                output.append(p.find('bndbox/ymin').text)
                output.append(p.find('bndbox/xmax').text)
                output.append(p.find('bndbox/ymax').text)

            print(output)
            f.write(','.join(output) + '\n')
