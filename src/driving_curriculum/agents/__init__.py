from .algorithmic import *
try:
    import tensorflow
    from .neural_networks.tf.end_to_end import *
    from .neural_networks.tf.goal_oriented import *
except RuntimeError:
    print('Could not find Tensorflow...')

