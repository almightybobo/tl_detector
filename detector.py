import tensorflow as tf
import numpy as np

from tensorflow.contrib import slim
from tensorflow.python.framework import graph_util

from data_aug import distort_image

class TrafficLightDetector:
  def __init__(self, input_shape, checkpoint, is_train, name="CBNOnet", **kwargs):
    self._input_shape = input_shape
    self._total_stride = 16
    self._output_shape = [input_shape[0], input_shape[1] // self._total_stride, input_shape[2] // self._total_stride, 4]
    self._checkpoint = checkpoint
    self._is_train = is_train
    self._name = name

    self._activation_fn = kwargs.get('activation_fn', tf.nn.relu)
    self._start_learning_rate = kwargs.get('start_learning_rate', 1e-3)
    self._lr_decay_steps = kwargs.get('lr_decay_steps', 100)
    self._lr_decay_rate = kwargs.get('lr_decay_rate', 0.96)
    self._lr_decay_staircase = kwargs.get('lr_decay_staircase', True)
    self._pos_thresh = kwargs.get('pos_thresh', 0.5)
    self._train_fast_mode = kwargs.get('train_fast_mode', False)

    self._session = None
    self._saver = None
    self._build()

  def __del__(self):
    self.close_session()

  @property
  def input_shape(self):
    return self._input_shape

  @property
  def output_shape(self):
    return self._output_shape

  @property
  def checkpoint(self):
    return self._checkpoint

  @property
  def is_train(self):
    return self._is_train

  @property
  def activation_fn(self):
    return self._activation_fn

  @property
  def name(self):
    return self._name

  @property
  def parameters_info(self):
    s = ""
    total_size = 0
    for v in tf.trainable_variables():
      size = np.prod(v.shape)
      s += "size: %d == %s\n" % (size, v)
      total_size += size
    s += "total_size: %d" % total_size
    return s

  @property
  def start_learning_rate(self):
    return self._start_learning_rate
  
  @property
  def lr_decay_steps(self):
    return self._lr_decay_steps

  @property
  def lr_decay_rate(self):
    return self._lr_decay_rate

  @property
  def lr_decay_staircase(self):
    return self._lr_decay_staircase

  @property
  def pos_thresh(self):
    return self._pos_thresh

  @property
  def sess(self):
    return self._session

  def create_session(self):
    self._session = tf.Session()

    if self.checkpoint is not None:
      saver = tf.train.Saver()
      saver.restore(self.sess, self.checkpoint)
      print('restore session from checkpoint %s' % self.checkpoint)
    else:
      self.sess.run(tf.global_variables_initializer())
      print('session initialized')

  def close_session(self):
    if self.sess is not None:
      self.sess.close()

  def save(self, save_path):
    if self._saver is None:
      self._saver = tf.train.Saver(max_to_keep=100)
    self._saver.save(self.sess, save_path)
  
  def save_pb(self, pb_path):
    extracted_graph = graph_util.extract_sub_graph(
        self.sess.graph_def,
        [self.name + '/light_state', self.name + '/light_position'])
    constant_graph = graph_util.convert_variables_to_constants(
        self.sess,
        self.sess.graph_def,
        [n.name for n in extracted_graph.node])
    for n in constant_graph.node:
      print(n.name)
    with tf.gfile.FastGFile(pb_path, mode='wb') as f:
      f.write(constant_graph.SerializeToString())

  def _build(self):
    self.node = {}
    with tf.variable_scope(self.name):
      self._build_model()
      if self.is_train:
        self._build_train()
        self._saver = tf.train.Saver(max_to_keep=10)

  def _data_aug(self, input_image):
    net = input_image
    net = tf.map_fn(lambda x: distort_image(x, fast_mode=self._train_fast_mode), net)
    return net

  def _build_model(self):
    ph_image = tf.placeholder(tf.float32, shape=self.input_shape, name='images')
    self.node['ph_image'] = ph_image

    net = ph_image
    net = net * (1. / 255.)
    if self.is_train:
      net = self._data_aug(net)

    with slim.arg_scope(
        [slim.conv2d],
        kernel_size=(3, 3), padding='SAME', normalizer_fn=slim.batch_norm, activation_fn=self.activation_fn):

      with slim.arg_scope(
          [slim.batch_norm], is_training=self.is_train):

        def resblock(net):
          in_channel = net.shape[-1]
          short_cut = net
          net = slim.conv2d(net, in_channel * 4, kernel_size=(1, 1))
          net = slim.separable_conv2d(net, in_channel, kernel_size=(3, 3), depth_multiplier=1)
          net = net + short_cut
          return net

        net = slim.conv2d(net, 32, stride=2)
        net = slim.conv2d(net, 32, stride=1)

        net = resblock(net)
        net = slim.conv2d(net, 32, stride=2)

        net = resblock(net)
        net = slim.conv2d(net, 64, stride=2)

        net = resblock(net)
        net = slim.conv2d(net, 64, stride=2)

        net = slim.conv2d(net, 4, stride=1, activation_fn=None, normalizer_fn=None)

    output = net
    self.node['output'] = output

    with tf.control_dependencies([tf.assert_equal(tf.shape(output)[0], 1)]):
      conf = output[0,:,:,0] # (H, W)
      conf = tf.nn.sigmoid(conf) # (H, W)
      pos = tf.greater_equal(conf, self.pos_thresh) # (H, W)
      pos_count = tf.reduce_sum(tf.cast(pos, tf.int8))
      
      logit = output[0,:,:,1:4] # (H, W, 3)
      logit = tf.boolean_mask(logit, pos) # (N, 3)
      light_states = tf.argmax(logit, -1) # (N)
      light_states, _, count = tf.unique_with_counts(light_states)
      max_index = tf.cond(
          tf.equal(pos_count, 0),
          lambda: tf.constant(0, dtype=tf.int32),
          lambda: tf.cast(tf.argmax(count, -1), tf.int32))
      light_state = tf.cond(
          tf.equal(pos_count, 0),
          lambda: tf.constant(-1, dtype=tf.int32),
          lambda: tf.cast(tf.gather(light_states, max_index), tf.int32))
      light_state = tf.identity(light_state, name='light_state')
      light_position = tf.where(pos) * self._total_stride + self._total_stride // 2
      light_position = tf.identity(light_position, name='light_position')

    self.node['light_state'] = light_state
    self.node['light_position'] = light_position

  def _build_train(self):
    output = self.node['output']

    batch_size = tf.shape(output)[0]
    batch_size = tf.cast(batch_size, tf.float32)
    ph_label = tf.placeholder(tf.float32, shape=self.output_shape)
    ph_mask = tf.placeholder(tf.float32, shape=self.output_shape)
    self.node['ph_label'] = ph_label
    self.node['ph_mask'] = ph_mask

    gt_conf = ph_label[:,:,:,0]
    pr_conf = output[:,:,:,0]
    mask_conf = ph_mask[:,:,:,0]
    # loss_conf = tf.squared_difference(gt_conf, pr_conf)
    pr_conf = tf.nn.sigmoid(pr_conf)
    pr_conf = tf.clip_by_value(pr_conf, 1e-3, 1-1e-3)
    loss_conf = - gt_conf * tf.log(pr_conf) - (1. - gt_conf) * tf.log(1. - pr_conf)
    loss_conf = loss_conf * mask_conf
    loss_conf = tf.reduce_sum(loss_conf) / batch_size
    self.node['loss_conf'] = loss_conf

    gt_logit = ph_label[:,:,:,1:4]
    pr_logit = tf.nn.softmax(output[:,:,:,1:4])
    pr_logit = tf.clip_by_value(pr_logit, 1e-3, 1-1e-3)
    mask_logit = ph_mask[:,:,:,1:4]
    loss_logit = - tf.reduce_sum(gt_logit * tf.log(pr_logit) * mask_logit, -1)
    loss_logit = tf.reduce_sum(loss_logit) / batch_size
    self.node['loss_logit'] = loss_logit

    loss = 5. * loss_conf + loss_logit
    self.node['loss'] = loss

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      global_step = tf.Variable(0, trainable=False)
      lr_decay = tf.train.exponential_decay(
          learning_rate=self.start_learning_rate,
          global_step=global_step,
          decay_steps=self.lr_decay_steps,
          decay_rate=self.lr_decay_rate,
          staircase=self.lr_decay_staircase)
      optimizer = tf.train.AdamOptimizer(lr_decay)
      train_step = optimizer.minimize(loss, global_step)
      self.node['global_step'] = global_step
      self.node['learning_rate'] = lr_decay
      self.node['train_step'] = train_step

  def train(self, images, labels, labels_mask):
    if np.random.random() > 0.5:
      images = images[:,:,::-1,:]
      labels = labels[:,:,::-1,:]
      labels_mask = labels_mask[:,:,::-1,:]
    ret = self.sess.run(
        {
            'step': self.node['global_step'],
            'lr': self.node['learning_rate'],
            'loss_conf': self.node['loss_conf'],
            'loss_logit': self.node['loss_logit'],
            'loss': self.node['loss'],
            'train_step': self.node['train_step']},
        {
            self.node['ph_image']: images,
            self.node['ph_label']: labels,
            self.node['ph_mask']: labels_mask})
    return ret

  def predict(self, images):
    ret = self.sess.run(
        {
            'light_state': self.node['light_state'],
            'light_position': self.node['light_position']},
        {
            self.node['ph_image']: images})
    return ret

  def test(self, images, labels, labels_mask):
    ret = self.sess.run(
        {
            'loss_conf': self.node['loss_conf'],
            'loss_logit': self.node['loss_logit'],
            'loss': self.node['loss']},
        {
            self.node['ph_image']: images,
            self.node['ph_label']: labels,
            self.node['ph_mask']: labels_mask})
    return ret

if __name__ == '__main__':
  def test():
    import cv2
    batch_size = 1
    tld = TrafficLightDetector(
        input_shape=[batch_size, 288, 384, 3],
        checkpoint=None,
        is_train=True)
    assert tld.input_shape == [batch_size, 288, 384, 3]
    assert tld.output_shape == [batch_size, 18, 24, 4]
    print(tld.parameters_info)

    train_txt_path = './udacity-traffic-light-dataset/train.txt'
    with open(train_txt_path, 'r') as f:
      for _ in range(868):
        sample = f.readline()
      sample = f.readline()
      sample = sample.split(',')
    # one example
    image_path = sample[0]
    labels = []
    boxes = []
    centers = []
    for i in range((len(sample) - 1) // 5):
      label = int(sample[i*5+1])
      x1 = float(sample[i*5+2])
      y1 = float(sample[i*5+3])
      x2 = float(sample[i*5+4])
      y2 = float(sample[i*5+5])
      center_x = (x1 + x2) / 2
      center_y = (y1 + y2) / 2
      labels.append(label)
      boxes.append([x1, y1, x2, y2])
      centers.append([center_x, center_y])

    label_kernel = np.array([[0.5, 0.7, 0.5], [0.5, 1.0, 0.5], [0.5, 0.7, 0.5]], dtype=np.float32)
    label = np.zeros(shape=[1, 18, 24, 4], dtype=np.float32)
    label_mask = np.zeros(shape=[1, 18, 24, 4], dtype=np.float32)
    label_mask[:, :, :, 0] = 0.1
    for center in centers:
      x = round(center[0] / 16) # 288 // 18 == 16
      x = np.clip(x, 0, label.shape[2] - 1)
      y = round(center[1] / 16) # 384 // 24 == 16
      y = np.clip(y, 0, label.shape[1] - 1)

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

      label[0, ly_s:ly_e+1, lx_s:lx_e+1, 0] = label_kernel[ky_s:ky_e+1, kx_s:kx_e+1]
      label_mask[0, ly_s:ly_e+1, lx_s:lx_e+1, :] = 1.

      for y in range(ly_s, ly_e+1):
        for x in range(lx_s, lx_e+1):
          label[0, y, x, labels[0]+1] = 1
      
    for c in range(4):
      print('\n')
      for y in range(18):
        print(label[0, y, :, c])
      print()
      for y in range(18):
        print(label_mask[0, y, :, c])
      
    image = np.expand_dims(cv2.imread(image_path), 0)
    tld.create_session()
    for _ in range(100):
      print(tld.train(image, label, label_mask))
      print(tld.predict(image))
    print(tld.test(image))
  test()