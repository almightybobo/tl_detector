import os
import argparse
import tensorflow as tf
import numpy as np
import cv2
from data_loader import DataLoader
from detector import TrafficLightDetector

def train(tld, data, args):
  epoch_steps = data.train_size // args.batch_size
  for e in range(1, args.epochs+1):
    print('---- start training ----')
    for step in range(1, epoch_steps+1):
      images, labels, labels_mask = data.get_train_batch()
      ret = tld.train(images, labels, labels_mask)
      print(ret, end='\r')
    print()
    ckpt_path = os.path.join(args.log_dir, 'ckpt-%d-%d' % (e, step))
    tld.save(ckpt_path)
    print('checkpoint %s saved' % ckpt_path)

    test(tld, data, args)
  
  print('End')

def test(tld, data, args):
  test_epoch_steps = data.test_size // args.batch_size
  print('---- start testing ----')
  for i in range(test_epoch_steps):
    images, labels, labels_mask = data.get_test_batch()
    ret = tld.test(images, labels, labels_mask)
    print(ret, end='\r')
  print()

def predict(tld, data, args):
  print('---- start predicting ----')
  correct = 0
  for i in range(data.test_size):
    example = data.get_one_test_example()
    image = cv2.imread(example.image_path)

    ret = tld.predict(np.expand_dims(image, 0))
    if ret['light_state'] == 0:
      color = (0, 0, 255)
    elif ret['light_state'] == 1:
      color = (255, 0, 0)
    else:
      color = (0, 255, 0)

    if ret['light_state'] == example.lights[0].light_state:
      correct += 1

    for y, x in ret['light_position']:
      cv2.circle(image, (x, y), 3, color, -1)

    output_path = os.path.join(args.log_dir, os.path.basename(example.image_path))
    cv2.imwrite(output_path, image)

  accuracy = correct / data.test_size
  print('accuracy: [%d/%d] = %.6f' % (correct, data.test_size, accuracy))

def main(args):
  tld = TrafficLightDetector(
      input_shape=[None, args.input_h, args.input_w, 3],
      checkpoint=args.ckpt,
      is_train=args.train or args.test,
      pos_thresh=args.pos_thresh)
  
  tld.create_session()
  
  data = DataLoader(
      args.data,
      args.batch_size,
      args.input_h,
      args.input_w,
      tld.output_shape[1],
      tld.output_shape[2],
      args.split_test)

  if args.train:
    train(tld, data, args)
  elif args.test:
    test(tld, data, args)
  elif args.predict:
    predict(tld, data, args)
  else:
    print("specity phase: -r, -t, -p")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  phase = parser.add_mutually_exclusive_group()
  phase.add_argument('-r', '--train', action='store_true', default=False)
  phase.add_argument('-t', '--test', action='store_true', default=False)
  phase.add_argument('-p', '--predict', action='store_true', default=False)

  parser.add_argument('-d', '--data', default='./udacity-traffic-light-dataset/train.txt', type=str)
  parser.add_argument('-c', '--ckpt', default=None)

  parser.add_argument('-b', '--batch_size', default=1, type=int)
  parser.add_argument('--input_h', default=288, type=int)
  parser.add_argument('--input_w', default=384, type=int)
  parser.add_argument('-s', '--split_test', default=0.1, type=float)

  parser.add_argument('-e', '--epochs', default=1, type=int)
  parser.add_argument('-l', '--log_dir', default='./logs', type=str)

  parser.add_argument('--pos_thresh', default=0.5, type=float)

  args = parser.parse_args()

  main(args)


