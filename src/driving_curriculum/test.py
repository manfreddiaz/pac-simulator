from curriculum_learning.config import *


def test():
    config = argument_parser('tester')
    print(config)

    learning_regime, storage_dir = create_world(config)

    learning_regime.test()


def test_teacher():
    config = argument_parser('tester')

    world, scenario, learner, teacher, storage_dir, stored_model = create_world(config)

    learning_regime = OnlineLaneFollowing(world=world, teacher=teacher, learner=learner, scenario=scenario,
                                          storage_location=stored_model)
    learning_regime.demo_teacher()


if __name__ == '__main__':
    test()

