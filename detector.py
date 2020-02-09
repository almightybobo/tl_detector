import tensorflow as tf
import numpy as np

from tensorflow.contrib import slim

class TrafficLightDetector:
  def __init__(self, input_shape, checkpoint, is_train, name="CBNOnet", **kwargs):
    self._input_shape = input_shape
    self._output_shape = [input_shape[0], input_shape[1] // 16, input_shape[2] // 16, 4]
    self._checkpoint = checkpoint
    self._is_train = is_train
    self._name = name

    self._activation_fn = kwargs.get('activation_fn', tf.nn.relu)
    self._start_learning_rate = kwargs.get('start_learning_rate', 1e-3)
    self._lr_decay_steps = kwargs.get('lr_decay_steps', 1000)
    self._lr_decay_rate = kwargs.get('lr_decay_rate', 0.96)
    self._lr_decay_staircase = kwargs.get('lr_decay_staircase', True)

    self._session = None
    self._build()

  def __del__(self):
    self.sess.close()

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
  def sess(self):
    return self._session

  def create_session(self):
    self._session = tf.Session()

    if self.checkpoint is not None:
      saver = tf.train.Saver()
      saver.restore(self.sess, self.checkpoint)
    else:
      self.sess.run(tf.global_variables_initializer())

  def close_session(self):
    if self.sess is not None:
      self.sess.close()

  def _build(self):
    self.node = {}
    with tf.variable_scope(self.name):
      self._build_model()
      if self.is_train:
        self._build_train()
        self._saver = tf.train.Saver(max_to_keep=10)

  def _build_model(self):
    ph_image = tf.placeholder(tf.float32, shape=self.input_shape)
    self.node['ph_image'] = ph_image

    # normalize ?!
    net = ph_image

    with slim.arg_scope(
        [slim.conv2d],
        kernel_size=(3, 3), padding='SAME', normalizer_fn=slim.batch_norm, activation_fn=self.activation_fn):

      with slim.arg_scope(
          [slim.batch_norm], is_training=self.is_train, scale=True):

        def resblock(net):
          in_channel = net.shape[-1]
          short_cut = net
          net = slim.conv2d(net, in_channel * 4, kernel_size=(1, 1))
          net = slim.separable_conv2d(net, in_channel, kernel_size=(3, 3))
          net = net + short_cut
          return net

        net = slim.conv2d(net, 32, stride=2)
        net = slim.conv2d(net, 32, stride=1)

        net = resblock(net)
        net = slim.conv2d(net, 32, stride=2)

        net = resblock(net)
        net = slim.conv2d(net, 32, stride=2)

        net = resblock(net)
        net = slim.conv2d(net, 32, stride=2)

        net = slim.conv2d(net, 4, stride=1, activation_fn=None, normalizer_fn=None)

    output = net
    self.node['output'] = output

  def _build_train(self):
    output = self.node['output']

    batch_size = tf.shape(output)[0]
    batch_size = tf.cast(batch_size, tf.float32)
    ph_label = tf.placeholder(tf.float32, shape=self.output_shape)
    ph_mask = tf.placeholder(tf.bool, shape=self.output_shape[:3])
    self.node['ph_label'] = ph_label
    self.node['ph_mask'] = ph_mask

    gt_conf = tf.boolean_mask(ph_label[:,:,:,0], ph_mask)
    pr_conf = tf.boolean_mask(output[:,:,:,0], ph_mask)
    pr_conf = tf.nn.sigmoid(pr_conf)
    loss_conf = tf.squared_difference(gt_conf, pr_conf)
    loss_conf = tf.reduce_sum(loss_conf) / batch_size
    self.node['loss_conf'] = loss_conf

    gt_logit = tf.boolean_mask(ph_label[:,:,:,1:4], ph_mask)
    pr_logit = tf.boolean_mask(output[:,:,:,1:4], ph_mask)
    loss_logit = tf.nn.softmax_cross_entropy_with_logits(labels=gt_logit, logits=pr_logit)
    loss_logit = tf.reduce_sum(loss_logit) / batch_size
    self.node['loss_logit'] = loss_logit

    loss = loss_conf + loss_logit
    self.node['loss'] = loss

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies([update_ops]):
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
    ret = self.sess.run(
        [
            self.node['global_step'],
            self.node['learning_rate'],
            self.node['loss_conf'],
            self.node['loss_logit'],
            self.node['loss'],
            self.node['train_step']],
        {
            self.node['ph_image']: images,
            self.node['ph_label']: labels,
            self.node['ph_mask']: labels_mask})
    return ret[:-1]

  def test(self, images):
    output = self.sess.run(self.node['output'], {self.node['ph_image']: images})
    return output

if __name__ == '__main__':
  def test():
    batch_size = 32
    tld = TrafficLightDetector(
        input_shape=[batch_size, 288, 384, 3],
        checkpoint=None,
        is_train=True)
    assert tld.input_shape == [batch_size, 288, 384, 3]
    assert tld.output_shape == [batch_size, 18, 24, 4]
    print(tld.parameters_info)

    tld.train()
    tld.test()
  test()