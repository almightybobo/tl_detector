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
  def __init__(self, light_state, box_center):
    self.light_state = light_state
    self.box_center = box_center

class Example:
  def __init__(self, image_path, lights):
    self.image_path = image_path
    self.lights = lights

class DataLoader:
  def __init__(self, train_txt_path, batch_size, input_h, input_w, output_h, output_w, test_ratio=0.1):
    examples = self._read_examples(train_txt_path)
    np.random.seed(84)
    np.random.shuffle(examples)
    self.train_size = int(round(len(examples) * (1 - test_ratio)))
    self.test_size = len(examples) - self.train_size
    self.train_examples = examples[:self.train_size]
    self.test_examples = examples[self.train_size:]

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

    self.images = np.zeros(shape=[batch_size, input_h, input_w, 3], dtype=np.float32)
    self.labels = np.zeros(shape=[batch_size, output_h, output_w, 4], dtype=np.float32)
    self.labels_mask = np.zeros(shape=[batch_size, output_h, output_w, 4], dtype=np.float32)

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
          light = Light(light_state, Point(center_x, center_y))
          lights.append(light)

        examples.append(Example(image_path, lights))
    return examples

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

  def transform(self, example, neg_coef=0.1, label_smooth=0.0):
    label = np.zeros(shape=[1, self.output_h, self.output_w, 4], dtype=np.float32)
    label_mask = np.zeros(shape=[1, self.output_h, self.output_w, 4], dtype=np.float32)
    label_mask[:, :, :, 0] = neg_coef

    for light in example.lights:
      
      x = int(round(light.box_center.x / self.stride))
      x = np.clip(x, 0, label.shape[2] - 1)
      y = int(round(light.box_center.y / self.stride))
      y = np.clip(y, 0, label.shape[1] - 1)

      label_mask[0, y, x, :] = 1.
      label[0, :, :, 0] = 0
      label[0, y, x, 0] = 1.
      label[0, y, x, light.light_state + 1] = 1. - label_smooth
      label[0, y, x, (light.light_state + 1) % 3 + 1] = label_smooth / 2
      label[0, y, x, (light.light_state + 2) % 3 + 1] = label_smooth / 2

      '''
      if x == 0:
        kx_s = 1
        kx_e = 2
        lx_s = 0
        lx_e = 1
      elif x == label.shape[2] - 1:
        kx_s = 0
        kx_e = 1
        lx_s = x - 1
        lx_e = x
      else:
        kx_s = 0
        kx_e = 2
        lx_s = x - 1
        lx_e = x + 1

      if y == 0:
        ky_s = 1
        ky_e = 2
        ly_s = 0
        ly_e = 1
      elif y == label.shape[1] - 1:
        ky_s = 0
        ky_e = 1
        ly_s = y - 1
        ly_e = y
      else:
        ky_s = 0
        ky_e = 2
        ly_s = y - 1
        ly_e = y + 1

      label[0, ly_s:(ly_e+1), lx_s:(lx_e+1), 0] = self.label_kernel[ky_s:(ky_e+1), kx_s:(kx_e+1)]
      label_mask[0, ly_s:(ly_e+1), lx_s:(lx_e+1), :] = 1.

      for y in range(ly_s, ly_e+1):
        for x in range(lx_s, lx_e+1):
          label[0, y, x, light.light_state + 1] = 1
      '''
      
    '''
    for c in range(4):
      print('\n')
      for y in range(18):
        print(label[0, y, :, c])
      print()
      for y in range(18):
        print(label_mask[0, y, :, c])
    '''
      
    image = np.expand_dims(cv2.imread(example.image_path), 0)
    return image, label, label_mask

if __name__ == '__main__':
  train_txt_path = './udacity-traffic-light-dataset/train.txt'
  loader = DataLoader(train_txt_path, 2, 288, 384, 18, 24)
  loader.get_train_batch()
  loader.get_test_batch()
