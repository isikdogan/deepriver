import tensorflow as tf

def model(input_layer, num_labels, training, num_filters=16, weight_decay=0.00001):

    reg = tf.contrib.layers.l2_regularizer(weight_decay)
    o_init = tf.orthogonal_initializer()

    def conv_block(inputs, name):
        with tf.variable_scope(name):
            net = tf.layers.separable_conv2d(
                        inputs=inputs,
                        filters=num_filters,
                        kernel_size=(3,3),
                        padding='SAME',
                        use_bias=False,
                        activation=None,
                        depthwise_initializer=o_init,
                        pointwise_initializer=o_init,
                        pointwise_regularizer=reg,
                        depthwise_regularizer=reg)
            net = tf.layers.batch_normalization(
                        inputs=net,
                        renorm=True,
                        training=training)
            net = tf.nn.relu(net)
        return net

    def pooling(inputs, name):
        with tf.variable_scope(name):
            net = tf.layers.max_pooling2d(inputs,
                                    pool_size=(2,2),
                                    strides=(2,2))
        return net

    def upsampling(inputs, name):
        with tf.variable_scope(name):
            dims = tf.shape(inputs)
            new_size = [dims[1]*2, dims[2]*2]
            net = tf.image.resize_bilinear(inputs, new_size)
        return net

    def pointwise_block(inputs, name):
        with tf.variable_scope(name):
            net = tf.layers.conv2d(
                        inputs=inputs,
                        filters=num_filters,
                        kernel_size=(1,1),
                        padding='SAME',
                        use_bias=False,
                        activation=None,
                        kernel_initializer=o_init,
                        kernel_regularizer=reg)
            net = tf.layers.batch_normalization(
                        inputs=net,
                        renorm=True,
                        training=training)
            net = tf.nn.relu(net)
        return net

    def output_block(inputs, num_labels, name):
        with tf.variable_scope(name):
            net = tf.layers.conv2d(
                        inputs=inputs,
                        filters=num_labels,
                        kernel_size=(1,1),
                        activation=None)
        return net

    def subnet_module(inputs, name, num_layers=3):
        with tf.variable_scope(name):
            for i in range(num_layers-1):
                net = conv_block(inputs, name='{}_conv{}'.format(name, i))
                inputs = tf.concat([net, inputs], axis=3)
            net = conv_block(inputs, name='{}_conv3'.format(name))
        return net

    scales = [] # use as a stack for FCN connections

    # entry flow
    with tf.variable_scope('entry_flow'):
        inputs = tf.expand_dims(input_layer, axis=3)
        net = conv_block(inputs, name='conv0')
        scales.append(net)

    # encoder
    with tf.variable_scope('encoder'):
        for i in range(8):
            pool = pooling(net, name='pool{}'.format(i))
            net = subnet_module(pool, name='subnet_e{}'.format(i))
            net = tf.concat([pool, net], axis=3)
            scales.append(net)

    # bottleneck
    with tf.variable_scope('bottleneck'):
        pool = pooling(net, name='pool_bottleneck')
        net = pointwise_block(pool, name='pointwise_bottleneck')

    # decoder
    with tf.variable_scope('decoder'):
        for i in range(8):
            upsample = upsampling(net, name='upsample{}'.format(i))
            net = tf.concat([upsample, scales.pop()], axis=3)
            net = subnet_module(net, name='subnet_d{}'.format(i))

    # exit flow
    with tf.variable_scope('exit_flow'):
        upsample = upsampling(net, name='upsample_exit')
        net = tf.concat([upsample, scales.pop()], axis=3)
        net = pointwise_block(net, name='pointwise_exit')
        logits = output_block(net, num_labels, name='output_block')

    return logits