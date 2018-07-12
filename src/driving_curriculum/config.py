import argparse

from .rendering import PacmanAvatarPyplot, LinePyplot
from simulator import VisualWorld, Engine
from .agents import TeacherPurePursuit, LearnerOneResidualMDN, LearnerOneResidualDropout, GoalOrientedLearnerOneResidualMDN, GoalOrientedLearnerOneResidualDropout, LearnerOneResidualRNNMDN, LearnerOneResidualRNNDropout
from .tasks.lane_following.scenarios.single_lane import RoadScenario, TIntersectionScenario,  YIntersectionScenario, HighwayExitsScenario
from .learning import GoalOrientedOnlineLaneFollowing, OnlineLaneFollowing

WORLD_WIDTH = 8  # [m]
WORLD_HEIGHT = 5  # [m]
LANE_WIDTH = 1.5  # [m]
LANE_THICKNESS = 1.5
LANE_LENGTH = 4   # [m]
LANE_Y = 2.125     # [m]
LANE_X = 0        # [m]

AGENT_RADIUS = (LANE_WIDTH - 0.7) / 2  # [m] see: european convention on Wikipedia

scenario_class = 'scenario_class_object'
scenario_name = 'scenario_name'

SCENARIOS = {
    'l': {
        scenario_class: RoadScenario,
        scenario_name: 'road_scenario'
    },
    'y': {
        scenario_class: YIntersectionScenario,
        scenario_name: 'y_intersection_scenario'
    },
    't': {
        scenario_class: TIntersectionScenario,
        scenario_name: 't_intersection_scenario'
    },
    'e': {
        scenario_class: HighwayExitsScenario,
        scenario_name: 'exits_scenario'
    },
}

learner_class = 'learner_class_object'
learner_name = 'learner_name'

LEARNERS = {
    'm': {
        learner_class: LearnerOneResidualMDN,
        learner_name: 'residual_mdn'
    },
    'd': {
        learner_class: LearnerOneResidualDropout,
        learner_name: 'residual_mcd'
    },
    'r': {
        learner_class: LearnerOneResidualRNNMDN,
        learner_name: 'residual_rnn_mdn'
    },
    'rd': {
        learner_class: LearnerOneResidualRNNDropout,
        learner_name: 'residual_rnn_mcd'
    },
    'gm': {
        learner_class: GoalOrientedLearnerOneResidualMDN,
        learner_name: 'goal_residual_mdn'
    },
    'gd': {
        learner_class: GoalOrientedLearnerOneResidualDropout,
        learner_name: 'goal_residual_mcd'
    }
}

teacher_class = 'teacher_class_object'
teacher_name = 'teacher_name'

TEACHERS = {
    'p': {
        teacher_class: TeacherPurePursuit,
        teacher_name: 'teacher_pure_pursuit'
    }
}

regime_class = 'regime_class_object'
regime_name = 'regime_name'

REGIMES = {
    'go': {
        regime_class: GoalOrientedOnlineLaneFollowing,
        regime_name: 'goal_online_lane'
    },
    'o': {
        regime_class: OnlineLaneFollowing,
        regime_name: 'online_lane'
    }
}


def _type_scenarios(scenario_arg):
    for scenario in SCENARIOS:
        if SCENARIOS[scenario][scenario_name] == scenario_arg:
            return SCENARIOS[scenario]


def _type_learner(learner_arg):
    for learner in LEARNERS:
        if LEARNERS[learner][learner_name] == learner_arg:
            return LEARNERS[learner]


def _type_teacher(teacher_arg):
    for teacher in TEACHERS:
        if TEACHERS[teacher][teacher_name] == teacher_arg:
            return TEACHERS[teacher]


def _type_regime(regime_arg):
    for regime in REGIMES:
        if REGIMES[regime][regime_name] == regime_arg:
            return REGIMES[regime]


def argument_parser(program):
    parser = argparse.ArgumentParser(prog=program, description='Single Lane Imitation Learning Scenarios')

    parser.add_argument('-s', '--scenario', choices=[scenario[scenario_name] for scenario in SCENARIOS.values()],
                        default='road_scenario')
    parser.add_argument('-l', '--learner', choices=[learner[learner_name] for learner in LEARNERS.values()],
                        default='residual_mdn')
    parser.add_argument('-t', '--teacher', choices=[teacher[teacher_name] for teacher in TEACHERS.values()],
                        default='teacher_pure_pursuit')
    parser.add_argument('-r', '--regime', choices=[regime[regime_name] for regime in REGIMES.values()],
                        default='online_lane')

    return parser.parse_args()


def create_world(config):
    world = VisualWorld(WORLD_WIDTH, WORLD_HEIGHT)
    world.set_engine(Engine())

    learner = _type_learner(config.learner)[learner_class](world)
    teacher = _type_teacher(config.teacher)[teacher_class](world)
    scenario = _type_scenarios(config.scenario)[scenario_class](LANE_X, LANE_Y, LANE_LENGTH, LANE_WIDTH, (WORLD_WIDTH, WORLD_HEIGHT))

    avatar = PacmanAvatarPyplot(world=world, radius=AGENT_RADIUS, color=(1.0, 0.83, 0))
    scenario_map = [LinePyplot(line=line, width=LANE_THICKNESS) for line in scenario.lines]

    world.visuals.extend(scenario_map)
    world.visuals.append(avatar)

    storage_dir = 'curriculum_learning/trained_models/{}/{}/{}'.format(
        config.scenario,
        config.teacher,
        config.learner,
    )
    stored_model = '{}/{}'.format(storage_dir, 'model')

    regime = _type_regime(config.regime)[regime_class](world, teacher, learner, scenario, stored_model)

    return regime, storage_dir
