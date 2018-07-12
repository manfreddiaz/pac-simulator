import tensorflow as tf

from ..layers import VariationalAutoencoder, VanillaAutoencoder, ConditionalVariationalAutoencoder
from ..tf_novelty_detector import TensorflowNoveltyDetector

tf.set_random_seed(1234)

STORAGE_DIR = "dt_learning/novelty/models/{}/"


class NoveltyDetectorAgent(TensorflowNoveltyDetector):

    def __init__(self, architecture, loss, filters_size, latent_size, learning, world, x=0.0, y=0.0, theta=0.0, v=0.0):
        TensorflowNoveltyDetector.__init__(self, world=world, learning=learning, x=x, y=y, theta=theta, v=v)
        self.arch = architecture
        self.loss = loss
        self.latent_size = latent_size
        self.filter_size = filters_size
        self.name = "{}_{}_conv_{}_lat_{}".format(architecture, loss, filters_size, latent_size)
        self.learning_rate = tf.Variable(initial_value=1e-3)

    def train(self, state_dims, action_dims, **kwargs):
        TensorflowNoveltyDetector.train(self, state_dims, action_dims, STORAGE_DIR.format(self.name))

    def test(self, state_dims, action_dims, **kwargs):
        TensorflowNoveltyDetector.test(self, state_dims, action_dims, STORAGE_DIR.format(self.name))

    def exploit(self, state, action, horizon=1):
        return TensorflowNoveltyDetector.exploit(self, state, action)

    def load_model(self):
        if self.arch == 'vae':
            return VariationalAutoencoder(self.state_tensor, filter_size=self.filter_size, latent_size=self.latent_size,
                                          loss_function=self.loss)
        elif self.arch == 'ae':
            return VanillaAutoencoder(self.state_tensor, filter_size=self.filter_size, loss_function=self.loss)
        elif self.arch == 'cvae':
            return ConditionalVariationalAutoencoder(self.state_tensor, self.action_tensor, filter_size=self.filter_size,
                                                     latent_size=self.latent_size, loss_function=self.loss)
        else:
            raise NotImplementedError()

    def loss_statistics(self, loss):
        # Knuth's formula for streamed mean and variance
        mean = tf.Variable(initial_value=0.0, trainable=False, name='loss_mean')
        non_normalized_variance = tf.Variable(initial_value=0.0, trainable=False, name='unnormalized_loss_variance')
        normalized_variance = tf.Variable(initial_value=0.0, trainable=False, name='loss_variance')

        mean_update = mean + tf.divide(loss - mean, tf.cast(self.global_step, dtype=tf.float32))
        non_normalized_variance_update = tf.add(non_normalized_variance, (loss - mean) * (loss - mean_update))
        normalized_variance_update = non_normalized_variance_update / (tf.cast(self.global_step, dtype=tf.float32) - 1.)

        update_op = [
            tf.cond(tf.logical_and(self.learning_tensor, self.global_step < 2),
                    true_fn=lambda: tf.assign(mean, loss),
                    false_fn=lambda: tf.assign(mean, mean_update)),
            tf.cond(tf.logical_and(self.learning_tensor, self.global_step < 2),
                    true_fn=lambda: tf.assign(non_normalized_variance, 0.0),
                    false_fn=lambda: tf.assign(non_normalized_variance, non_normalized_variance_update)),
            tf.cond(tf.logical_and(self.learning_tensor, self.global_step < 2),
                    true_fn=lambda: tf.assign(normalized_variance, 0.0),
                    false_fn=lambda: tf.assign(normalized_variance, normalized_variance_update))
        ]

        with tf.name_scope('statistics'):
            tf.summary.scalar('avg_mean', mean)
            tf.summary.scalar('avg_variance', normalized_variance)

        return mean, normalized_variance, update_op

    def architecture(self):
        model = self.load_model()
        # mean, normalized_variance, update_op = self.loss_statistics(model.loss)

        if self.learning:
            # return [model.decoder, update_op], model.loss
            return [model.decoder], model.loss
        else:
            # return [tf.norm(model.loc), mean, normalized_variance], model.loss
            return [tf.norm(model.loc), tf.reduce_prod(model.latent.variance(), axis=1)], model.loss

    def get_optimizer(self, loss):
        with tf.name_scope('optimization'):
            tf.summary.scalar('learning_rate', self.learning_rate)
        return tf.train.AdamOptimizer(self.learning_rate).minimize(loss, global_step=self.global_step)

    def execute(self, action):
        raise NotImplementedError()
