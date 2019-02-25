import tensorflow as tf
def convBlock(inputs,numOut,name="convBlock"):
    with tf.name_scope(name):
        norm_1 = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu)
        conv_1 = tf.layers.conv2d(norm_1, int(numOut / 2), 1, (1, 1), padding="SAME")
        norm_2 = tf.contrib.layers.batch_norm(conv_1, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu)
        conv_2 = tf.layers.conv2d(norm_2, int(numOut / 2), 3, (1, 1), padding="SAME")
        norm_3 = tf.contrib.layers.batch_norm(conv_2, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu)
        conv_3 = tf.layers.conv2d(norm_3, int(numOut), 1, (1, 1), padding="SAME")
        return conv_3
def skiplayer(inputs,numOut,name="skiplayer"):
    with tf.name_scope(name):
        if inputs.get_shape().as_list()[3] == numOut:
            return inputs
        else:
            conv = tf.layers.conv2d(inputs, int(numOut), 1, (1, 1), padding="SAME")
            return conv
def residual(inputs,numOut,name="residual"):
    with tf.name_scope(name):
        convb = convBlock(inputs, numOut)
        skip = skiplayer(inputs, numOut)
        return tf.add_n([convb,skip])

def hourgalss(input,name):
    with tf.variable_scope(name + 'Hourgalss', reuse=tf.AUTO_REUSE):
        net1 = residual(input, 64, name="Net1")
        net2 = tf.layers.max_pooling2d(net1, 2, 2, padding="SAME", name="Net2")
        net3=residual(net2, 128, name="Net3")
        net4 = tf.layers.max_pooling2d(net3, 2, 2, padding="SAME", name="Net4")
        net5=residual(net4,256,name="Net5")
        net6 = tf.layers.max_pooling2d(net5, 2, 2, padding="SAME", name="Net6")
        net7=residual(net6,512,name="Net7")
        net8 = tf.layers.max_pooling2d(net7, 2, 2, padding="SAME", name="Net8")
        net9 = residual(net8, 512, name="Net9")
        net10 = residual(net9, 512, name="Net10")

        net7_=residual(net7, 512, name="Net7_")
        net11 = tf.image.resize_nearest_neighbor(net10, tf.shape(net10)[1:3] * 2, name="Net11")
        net12=tf.add_n([net11,net7_],name="Net11")
        net13=residual(net12, 256, name="Net13")

        net5_=residual(net5,256, name="Net5_")
        net14 = tf.image.resize_nearest_neighbor(net13, tf.shape(net13)[1:3] * 2, name="Net14")
        net15 = tf.add_n([net14, net5_], name="Net15")
        net16=residual(net15, 128, name="Net16")

        net3_ = residual(net3, 128, name="Net3_")
        net17 = tf.image.resize_nearest_neighbor(net16, tf.shape(net16)[1:3] * 2, name="Net17")
        net18 = tf.add_n([net17, net3_], name="Net18")
        net19 = residual(net18, 64, name="Net19")

        net1_ = residual(net1, 64, name="Net1_")
        net20 = tf.image.resize_nearest_neighbor(net19, tf.shape(net19)[1:3] * 2, name="Net20")
        net21 = tf.add_n([net20, net1_], name="Net21")
        net22 = residual(net21, 64, name="Net22")
        print(net22)
    return net22
def add_stage(input1,input2,input3,name="add_stage"):
    with tf.name_scope(name):
        conv_1 = tf.layers.conv2d(input1,64,1,(1,1))
        norm_1 = tf.contrib.layers.batch_norm(conv_1, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu)
        norm_2 = tf.contrib.layers.batch_norm(input2, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu)
        res1 = residual(norm_2, 64)
        res2=residual(res1,64)
        out=tf.add_n([norm_1,res2,input3])
    return out

class Generator:
  def __init__(self, name):
    self.name = name
    self.reuse = False
  def __call__(self, input):
    """
    Args:
      input: batch_size x image_size x image_size x 3
    Returns:
      output: 4D tensor batch_size x out_size x out_size x 1 (default 1x5x5x1)
              filled with 0.9 if real, 0.0 if fake
    """
    with tf.variable_scope(self.name):
      # convolution layers
      conv = tf.layers.conv2d(input["x"], 256, 7, (1, 1), padding="SAME", name="conv1", activation=None)
      norm = tf.contrib.layers.batch_norm(conv, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu)
      res1 = residual(norm, 128)
      pool = tf.layers.max_pooling2d(res1, 2, (2, 2), padding="SAME")
      res2 = residual(pool, 256)

      stage1 = hourgalss(res2, "Stage1")
      loss1 = tf.layers.conv2d(stage1, 13, 1, (1, 1), padding="SAME")
      stage1_1 = tf.layers.conv2d(loss1, 64, 1, (1, 1), padding="SAME")
      stage1_1 = tf.contrib.layers.batch_norm(stage1_1, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu)
      out1 = add_stage(res2, stage1, stage1_1, name="out1")

      stage2 = hourgalss(out1, "Stage2")
      loss2 = tf.layers.conv2d(stage2, 13, 1, (1, 1), padding="SAME")
      print(loss2)
      losses1 = tf.stack([loss1, loss2], 0)

      image1 = tf.add_n([tf.expand_dims(loss2[:, :, :, 0], -1), tf.expand_dims(loss2[:, :, :, 1], -1),
                         tf.expand_dims(loss2[:, :, :, 2], -1), tf.expand_dims(loss2[:, :, :, 3], -1)
                            , tf.expand_dims(loss2[:, :, :, 4], -1), tf.expand_dims(loss2[:, :, :, 5], -1),
                         tf.expand_dims(loss2[:, :, :, 6], -1),
                         tf.expand_dims(loss2[:, :, :, 7], -1), tf.expand_dims(loss2[:, :, :, 8], -1)
                            , tf.expand_dims(loss2[:, :, :, 9], -1), tf.expand_dims(loss2[:, :, :, 10], -1),
                         tf.expand_dims(loss2[:, :, :, 11], -1),
                         tf.expand_dims(loss2[:, :, :, 12], -1)])
      print(image1)
      stage3 = hourgalss(image1, "Stage3")
      loss3 = tf.layers.conv2d(stage3, 68, 1, (1, 1), padding="SAME")
      stage3_1 = tf.layers.conv2d(loss3, 64, 1, (1, 1), padding="SAME")
      stage3_1 = tf.contrib.layers.batch_norm(stage3_1, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu)
      out3 = add_stage(image1, stage3, stage3_1, name="out3")

      stage4 = hourgalss(out3, "Stage4")
      loss4 = tf.layers.conv2d(stage4, 68, 1, (1, 1), padding="SAME")
      losses2 = tf.stack([loss3, loss4], 0)
    self.image=image1
    self.reuse = True
    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
    return loss1,losses2

class Discriminator:
  def __init__(self, name, is_training=True, norm='instance', use_sigmoid=False):
    self.name = name
    self.is_training = is_training
    self.norm = norm
    self.reuse = False
    self.use_sigmoid = use_sigmoid

  def __call__(self, input):
    """
    Args:
      input: batch_size x image_size x image_size x 3
    Returns:
      output: 4D tensor batch_size x out_size x out_size x 1 (default 1x5x5x1)
              filled with 0.9 if real, 0.0 if fake
    """
    with tf.variable_scope(self.name):
      # convolution layers
      C64 = Ck(input, 64, reuse=self.reuse, norm=None,
          is_training=self.is_training, name='C64')             # (?, w/2, h/2, 64)
      C128 = Ck(C64, 128, reuse=self.reuse, norm=self.norm,
          is_training=self.is_training, name='C128')            # (?, w/4, h/4, 128)
      C256 = Ck(C128, 256, reuse=self.reuse, norm=self.norm,
          is_training=self.is_training, name='C256')            # (?, w/8, h/8, 256)
      C512 = Ck(C256, 512,reuse=self.reuse, norm=self.norm,
          is_training=self.is_training, name='C512')            # (?, w/16, h/16, 512)

      # apply a convolution to produce a 1 dimensional output (1 channel?)
      # use_sigmoid = False if use_lsgan = True
      flatten=tf.layers.flatten(C512)
      output = tf.layers.dense(flatten,1,activation=tf.nn.sigmoid,reuse=self.reuse,name='output')
      print(output)
    self.reuse = True
    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
    return output
def _weights(name, shape, mean=0.0, stddev=0.02):
  var = tf.get_variable(
    name, shape,
    initializer=tf.random_normal_initializer(
      mean=mean, stddev=stddev, dtype=tf.float32))
  return var
def _norm(input, is_training, norm='instance'):
  if norm == 'instance':
    return _instance_norm(input)
  elif norm == 'batch':
    return _batch_norm(input, is_training)
  else:
    return input
def _batch_norm(input, is_training):
  with tf.variable_scope("batch_norm"):
    return tf.contrib.layers.batch_norm(input,
                                        decay=0.9,
                                        scale=True,
                                        updates_collections=None,
                                        is_training=is_training)

def _instance_norm(input):
  with tf.variable_scope("instance_norm"):
    depth = input.get_shape()[3]
    scale = _weights("scale", [depth], mean=1.0)
    offset = _biases("offset", [depth])
    mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
    epsilon = 1e-5
    inv = tf.rsqrt(variance + epsilon)
    normalized = (input-mean)*inv
    return scale*normalized + offset
def _biases(name, shape, constant=0.0):
  return tf.get_variable(name, shape,
            initializer=tf.constant_initializer(constant))
def _leaky_relu(input, slope):
  return tf.maximum(slope*input, input)
def Ck(input, k, slope=0.2, stride=2, reuse=False, norm='instance', is_training=True, name=None):
  with tf.variable_scope(name, reuse=reuse):
    weights = _weights("weights",
      shape=[4, 4, input.get_shape()[3], k])

    conv = tf.nn.conv2d(input, weights,
        strides=[1, stride, stride, 1], padding='SAME')

    normalized = _norm(conv, is_training, norm)
    output = _leaky_relu(normalized, slope)
    return output
def last_conv(input, reuse=False, use_sigmoid=False, name=None):
  with tf.variable_scope(name, reuse=reuse):
    weights = _weights("weights",
      shape=[4, 4, input.get_shape()[3], 1])
    biases = _biases("biases", [1])

    conv = tf.nn.conv2d(input, weights,
        strides=[1, 1, 1, 1], padding='SAME')
    output = conv + biases
    if use_sigmoid:
      output = tf.sigmoid(output)
    return output









