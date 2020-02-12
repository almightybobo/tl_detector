from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import numpy as np
import cv2


class Point:
  def __init__(self, x, y):
    self.x = x
    self.y = y

class Light:
  def __init__(self, light_state, box_center, box_length):
    self.light_state = light_state
    self.box_center = box_center
    self.box_length = box_length

class Example:
  def __init__(self, image_path, lights):
    self.image_path = image_path
    self.lights = lights

class DataLoader:
  def __init__(self, train_txt_path, batch_size, input_h, input_w, output_h, output_w, n_group, test_ratio=0.1):
    examples = self._read_examples(train_txt_path)

    np.random.seed(84)
    np.random.shuffle(examples)
    self.train_size = int(round(len(examples) * (1 - test_ratio)))
    self.test_size = len(examples) - self.train_size
    self.train_examples = examples[:self.train_size]
    self.test_examples = examples[self.train_size:]

    self.stat = self.get_stat(self.train_examples)

    self.start = 0
    self.test_start = 0
    self.batch_size = batch_size
    self.input_h = input_h
    self.input_w = input_w
    self.output_h = output_h
    self.output_w = output_w
    assert input_h // output_h == input_w // output_w
    self.stride = input_h // output_h

    self.label_kernel = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0]], dtype=np.float32)

    self.n_group = 3
    self.ch_per_group = 4
    self.output_channel = self.ch_per_group * self.n_group
    self.images = np.zeros(shape=[batch_size, input_h, input_w, 3], dtype=np.float32)
    self.labels = np.zeros(shape=[batch_size, output_h, output_w, self.output_channel], dtype=np.float32)
    self.labels_mask = np.zeros(shape=[batch_size, output_h, output_w, self.output_channel], dtype=np.float32)

  def _read_examples(self, train_txt_path):
    examples = []
    with open(train_txt_path, 'r') as f:
      for sample in f:
        sample = sample.split(',')

        image_path = sample[0]
        lights = []
        for i in range((len(sample) - 1) // 5):
          light_state = int(sample[i*5+1])
          x1 = float(sample[i*5+2])
          y1 = float(sample[i*5+3])
          x2 = float(sample[i*5+4])
          y2 = float(sample[i*5+5])
          center_x = (x1 + x2) / 2
          center_y = (y1 + y2) / 2
          length_x = abs(x2 - x1)
          length_y = abs(y2 - y1)
          light = Light(light_state, Point(center_x, center_y), Point(length_x, length_y))
          lights.append(light)
        examples.append(Example(image_path, lights))
    return examples
  
  def get_stat(self, examples: [Example]):
    length = []
    area = []
    for example in examples:
      for light in example.lights:
        length.append([light.box_length.x, light.box_length.y])
        area.append(light.box_length.x * light.box_length.y)

    stat = {
        'length_mean': np.mean(length, 0),
        'length_q1': np.quantile(length, 0.25, 0),
        'length_q2': np.quantile(length, 0.5, 0),
        'length_q3': np.quantile(length, 0.75, 0),
        'area_mean': np.mean(area),
        'area_q1': np.quantile(area, 0.25),
        'area_q2': np.quantile(area, 0.5),
        'area_q3': np.quantile(area, 0.75)}

    print("Box stat:")
    for key, value in stat.items():
      print(key, value)
    return stat
     
  def get_train_batch(self):
    for i in range(self.batch_size):
      image, label, label_mask = self.get_one_train()
      self.images[i:(i+1)] = image
      self.labels[i:(i+1)] = label
      self.labels_mask[i:(i+1)] = label_mask
    return self.images, self.labels, self.labels_mask

  def get_test_batch(self):
    for i in range(self.batch_size):
      image, label, label_mask = self.get_one_train()
      self.images[i:(i+1)] = image
      self.labels[i:(i+1)] = label
      self.labels_mask[i:(i+1)] = label_mask
    return self.images, self.labels, self.labels_mask

  def get_one_train_example(self):
    if self.start == len(self.train_examples):
      np.random.shuffle(self.train_examples)
      self.start = 0
    
    example = self.train_examples[self.start]
    self.start += 1
    return example

  def get_one_train(self):
    example = self.get_one_train_example()
    return self.transform(example)

  def get_one_test_example(self):
    if self.test_start == len(self.test_examples):
      self.test_start = 0

    example = self.test_examples[self.test_start]
    self.test_start += 1
    return example

  def get_one_test(self):
    example = self.get_one_test_example()
    return self.transform(example)

  def transform(self, example, neg_coef=0.15, label_smooth=0.0):
    label = np.zeros(shape=[1, self.output_h, self.output_w, self.output_channel], dtype=np.float32)
    label_mask = np.zeros(shape=[1, self.output_h, self.output_w, self.output_channel], dtype=np.float32)
    label_mask[:, :, :, ::self.ch_per_group] = neg_coef

    for light in example.lights:
      
      x = int(round(light.box_center.x / self.stride))
      if x < 0 or x > label.shape[2] - 1:
        continue
      y = int(round(light.box_center.y / self.stride))
      if y < 0 or y > label.shape[1] - 1:
        continue

      area = light.box_length.x * light.box_length.y
      if area < self.stat['area_q1']:
        groups = [0]
      elif area < self.stat['area_q2']:
        groups = [0, 1]
      elif area < self.stat['area_q3']:
        groups = [1, 2]
      else:
        groups = [2]

      for g in groups:
        ch = g * self.ch_per_group
        label_mask[0, y, x, ch:(ch+self.ch_per_group)] = 1.
        label[0, y, x, ch] = 1.
        label[0, y, x, ch + light.light_state + 1] = 1.

    image = np.expand_dims(cv2.imread(example.image_path), 0)
    return image, label, label_mask

if __name__ == '__main__':
  train_txt_path = './udacity-traffic-light-dataset/train.txt'
  loader = DataLoader(train_txt_path, 1, 288, 384, 18, 24, 3)
  image, label, label_mask = loader.get_train_batch()
  for c in range(3*4):
    print(c)
    for y in range(18):
      if np.any(label[0, y, :, c]):
        print(label[0, y, :, c], ' v')
      else:
        print(label[0, y, :, c])

      if c % 4 == 0:
        if np.any(label_mask[0, y, :, c] != 0.15):
          print(' ', label_mask[0, y, :, c], ' v')
        else:
          print(' ', label_mask[0, y, :, c])
      else:
        if np.any(label_mask[0, y, :, c]):
          print(' ', label_mask[0, y, :, c], ' v')
        else:
          print(' ', label_mask[0, y, :, c])

    print()

  loader.get_test_batch()
