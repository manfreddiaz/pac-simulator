import tensorflow as tf


from ..layers import resnet_1
from ..tf_online_learner import TensorflowOnlineLearner

tf.set_random_seed(1234)


class LearnerOneResidualRNNDropout(TensorflowOnlineLearner):
    def __init__(self, world, x=0.0, y=0.0, theta=0.0, v=0.0, name=None):
        self.name = name
        TensorflowOnlineLearner.__init__(self, world, x=x, y=y, theta=theta, v=v)

    def exploit(self, state, horizon=1):
        regression = TensorflowOnlineLearner.exploit(self, state)
        return regression

    def architecture(self):
        model = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), self.state_tensor)
        model = resnet_1(model)
        model = tf.layers.dense(model, units=64, activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))

        lstm = tf.nn.rnn_cell.BasicLSTMCell(32, state_is_tuple=True)
        initial_state = lstm.zero_state(1, dtype=tf.float32)
        model, states = tf.nn.static_rnn(lstm, [model], initial_state=initial_state, dtype=tf.float32)
        model = tf.transpose(model, [1, 0, 2])
        #
        model = tf.layers.dense(model, 1)
        # model = tf.reshape(model, shape=(1, ))

        loss = tf.losses.mean_squared_error(model, self.action_tensor)
        tf.summary.scalar('loss', loss)

        return model, loss

    def get_optimizer(self, loss):
        return tf.train.AdagradOptimizer(1e-3).minimize(loss, global_step=self.global_step)

