import tensorflow as tf

from ..layers import resnet_1, MixtureDensityNetwork
from ..tf_online_learner import TensorflowOnlineLearner

tf.set_random_seed(1234)


class LearnerOneResidualMDN(TensorflowOnlineLearner):
    def __init__(self, world, x=0.0, y=0.0, theta=0.0, v=0.0):
        TensorflowOnlineLearner.__init__(self, world, x=x, y=y, theta=theta, v=v)

    def exploit(self, state, horizon=1):
        mdn = TensorflowOnlineLearner.exploit(self, state)
        # print('prediction')
        # print(mdn)
        return MixtureDensityNetwork.max_central_value(mdn[0], mdn[1], mdn[2])

    def architecture(self):
        model = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), self.state_tensor)
        model = resnet_1(model, keep_prob=1.0)
        # TRY: tanh + 64 or change to (relu, crelu), without dense
        model = tf.layers.dense(model, units=128, activation=tf.nn.tanh,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))

        loss, components, _ = MixtureDensityNetwork.create(model, self.action_tensor, number_mixtures=5)
        return components, loss

    def get_optimizer(self, loss):
        return tf.train.AdagradOptimizer(1e-3).minimize(loss, global_step=self.global_step)

