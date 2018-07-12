from math import cos, sin
import numpy as np
import tensorflow as tf

from .....simulator import Agent
# from simulator import Agent

tf.set_random_seed(1234)


class TensorflowTrainer(Agent):
    def __init__(self, world, x=0.0, y=0.0, theta=0.0, v=0.0, checkpoint_file=None):
        Agent.__init__(self, world, x, y, theta, v)
        self.state_tensor = None
        self.action_tensor = None
        self.policy_model = None
        self.optimization_algorithm = None
        self.loss_function = None
        self.last_loss = None
        self.tf_session = tf.InteractiveSession()
        self.tf_checkpoint = checkpoint_file
        self.tf_saver = None
        self.summary_merge = None
        self.summary_writer = None
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

    def exploit(self, state, horizon=1):
        action = self.tf_session.run([self.policy_model], feed_dict={
            self.state_tensor: [state],
        })
        action = np.squeeze(action)
        return action

    def learn(self, state, action):
        summary, step, _, learning_loss = self.tf_session.run(
            [self.summary_merge, self.global_step, self.optimization_algorithm, self.loss_function],
            feed_dict={
                self.state_tensor: [state],
                self.action_tensor: action
            }
        )
        self.summary_writer.add_summary(summary, step)
        self.last_loss = learning_loss
        return learning_loss

    def execute(self, action):
        self.theta = action
        self.x = self.x + self.v * cos(self.theta)
        self.y = self.y + self.v * sin(self.theta)

    def explore(self, state, horizon=1):
        pass

    def commit(self):
        self.tf_saver.save(self.tf_session, self.tf_checkpoint, global_step=self.global_step)

    def architecture(self):
        raise NotImplementedError()

    def train(self, state_dims, action_dims, storage_location):
        if not self.policy_model:
            self._state_action_tensors(state_dims, action_dims)
            self.policy_model, self.loss_function = self.architecture()
            self.optimization_algorithm = self.get_optimizer(self.loss_function)
            self.tf_session.run(tf.global_variables_initializer())
            tf.train.global_step(self.tf_session, self.global_step)
            self.summary_merge = tf.summary.merge_all()
            self.last_loss = float('inf')
            self.tf_checkpoint = tf.train.latest_checkpoint(storage_location)
            self.tf_saver = tf.train.Saver(filename='model')
            if self.tf_checkpoint:
                self.tf_saver.restore(self.tf_session, self.tf_checkpoint)
            else:
                self.tf_checkpoint = storage_location + 'model'
            self.summary_writer = tf.summary.FileWriter(storage_location, self.tf_session.graph)

    def test(self, state_dims, action_dims, storage_location):
        if not self.policy_model:
            self._state_action_tensors(state_dims, action_dims)
            self.policy_model, self.loss_function = self.architecture()
            self.tf_session.run(tf.global_variables_initializer())
            self.tf_checkpoint = tf.train.latest_checkpoint(storage_location)
            self.tf_saver = tf.train.Saver()
            if self.tf_checkpoint:
                self.tf_saver.restore(self.tf_session, self.tf_checkpoint)
            else:
                print("NO TRAINING!")

    def _state_action_tensors(self, input_shape=(None, 1), output_shape=(1, 2)):
        if len(input_shape) == 3:
            input_shape = (1, input_shape[0], input_shape[1], input_shape[2])
        if len(output_shape) == 0:
            output_shape = (1, 2)
        with tf.name_scope('data'):
            self.state_tensor = tf.placeholder(dtype=tf.float32, shape=input_shape, name='state')
            tf.summary.image('state', self.state_tensor, 1)
            self.action_tensor = tf.placeholder(dtype=tf.float32, shape=output_shape, name='action')
            tf.summary.scalar('action', tf.reshape(self.action_tensor, shape=[]))

    def get_optimizer(self, loss):
        raise NotImplementedError()
