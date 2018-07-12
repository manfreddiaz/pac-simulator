import os
import tensorflow as tf
from curriculum_learning.config import *

tf.set_random_seed(1234)


def _launch_tensorboard(running_dir):
    import os
    os.system('tensorboard --logdir={}'.format(running_dir))
    return


def _debug(running_dir):
    import threading
    debug_thread = threading.Thread(target=_launch_tensorboard, args=([running_dir]))
    debug_thread.start()
    return debug_thread


def train(a):
    config = argument_parser('trainer')
    learning_regime, storage_dir = create_world(config)

    os.makedirs(storage_dir, exist_ok=True)

    d_thread = _debug(storage_dir)
    print('Training...{}, {}, {}'.format(config.scenario, config.teacher, config.learner))

    learning_regime.train()
    d_thread.join()
    print('Training ended.')


if __name__ == '__main__':
    tf.app.run(main=train)






