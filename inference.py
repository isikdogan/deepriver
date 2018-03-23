import tensorflow as tf
import cv2
import numpy as np
from nets import model

class TFInference:

    def __init__(self, checkpoint_path='./checkpoints/'):
        self.checkpoint_path = checkpoint_path
        self.num_labels = 3

        self.input_layer = tf.placeholder(tf.float32, shape=[None, None], name='input')
        logits = model(tf.expand_dims(self.input_layer, axis=0),
                        num_labels=self.num_labels, training=False)
        self.preds = tf.squeeze(tf.nn.softmax(logits))

    def run(self, image_path, save_path):
        image = cv2.imread(image_path, 0)
        image = image.astype(np.float32)
        image = image / 255

        # don't allocate entire gpu memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            saver = tf.train.Saver(max_to_keep=None)
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            segmentation_map = sess.run(self.preds, feed_dict={self.input_layer: image})

        cv2.imwrite(save_path,(segmentation_map[:,:,1] * 255).astype(np.uint8))

def main():
    model = TFInference()
    model.run('./data/synthetic_input.jpg', 'data/synthetic_input_centerlines.png')
    model.run('./data/natural_input.jpg', 'data/natural_input_centerlines.png')

if __name__ == '__main__':
    main()