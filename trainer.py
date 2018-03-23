import tensorflow as tf
import os
from nets import model
from datagenerator import DataGenerator

class TFModelTrainer:

    def __init__(self, checkpoint_path='./checkpoints/'):
        self.checkpoint_path = checkpoint_path

        # set training parameters
        self.learning_rate = 0.0001
        self.num_iter = 1000000
        self.log_iter = 500
        self.save_iter = 5000
        self.batch_size = 1 # use batch re-norm which allows batch_size = 1

        # create the data generator
        self.image_size = (1024, 1024)
        self.data_generator = DataGenerator(self.image_size)
        self.num_labels = 3

    def _parse_function(self, input, labels, widthmap):
        ''' allows for parallel calls'''
        return input, labels, widthmap

    def _data_layer(self, num_threads=8, prefetch_buffer=64):
        with tf.name_scope('data'):
            dataset = tf.data.Dataset.from_generator(lambda: self.data_generator,
                                                     (tf.float32, tf.int32, tf.int32),
                                                     (tf.TensorShape(self.image_size),
                                                      tf.TensorShape(self.image_size),
                                                      tf.TensorShape(self.image_size)))
            dataset = dataset.map(self._parse_function, num_parallel_calls=num_threads)
            dataset = dataset.prefetch(prefetch_buffer)
            dataset = dataset.batch(self.batch_size)
            iterator = dataset.make_one_shot_iterator()

        return iterator

    def _loss_functions(self, logits, labels):
        with tf.name_scope('loss'):
            target_prob = tf.one_hot(labels, depth=self.num_labels)

            # max pooled loss
            xent = tf.nn.softmax_cross_entropy_with_logits(labels=target_prob, logits=logits)
            xent = tf.expand_dims(xent, 3)
            xent = tf.nn.max_pool(xent, ksize=[1,8,8,1], strides=[1,8,8,1], padding='SAME')
            xent = tf.reduce_mean(xent)
            tf.add_to_collection(tf.GraphKeys.LOSSES, xent)

            total_loss = tf.losses.get_total_loss() #include regularization loss

        return total_loss

    def _optimizer(self, total_loss, global_step):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=0.1)
            optimizer = optimizer.minimize(total_loss, global_step=global_step)

        return optimizer

    def _summaries(self, logits):
        # training summaries
        mean_loss = tf.placeholder(tf.float32)
        train_summaries = [tf.summary.scalar('mean_loss', mean_loss)]
        summ_op = tf.summary.merge(train_summaries)
        return summ_op, mean_loss

    def train(self):
        # iteration number
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        # training graph
        iterator = self._data_layer()
        input_layer, labels, _ = iterator.get_next()
        logits = model(input_layer, self.num_labels, training=True)
        total_loss = self._loss_functions(logits, labels)
        optimizer = self._optimizer(total_loss, global_step)

        # summary ops and placeholders
        summ_op, mean_loss = self._summaries(logits)

        # don't allocate entire gpu memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            writer = tf.summary.FileWriter(self.checkpoint_path, sess.graph)

            saver = tf.train.Saver(max_to_keep=None) # keep all checkpoints
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)

            # resume training if a checkpoint exists
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            initial_step = global_step.eval()

            # train the model
            streaming_loss = 0
            for i in range(initial_step, self.num_iter + 1):
                _, loss_batch = sess.run([optimizer, total_loss])

                # log training statistics
                streaming_loss += loss_batch
                if i % self.log_iter == self.log_iter - 1:
                    streaming_loss /= self.log_iter
                    print(i + 1, streaming_loss)
                    summary = sess.run(summ_op, feed_dict={mean_loss: streaming_loss})
                    writer.add_summary(summary, global_step=i)
                    streaming_loss = 0

                # save model
                if i % self.save_iter == self.save_iter - 1:
                    saver.save(sess, os.path.join(self.checkpoint_path, 'checkpoint'), global_step=global_step)
                    print("Model saved!")

            writer.close()

def main():
    trainer = TFModelTrainer()
    trainer.train()

if __name__ == '__main__':
    main()