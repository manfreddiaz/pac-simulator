from simulator import VisualWorld, Engine
from curriculum_learning.rendering import PacmanAvatarPyplot, LinePyplot, DirectionSignPyplot
from curriculum_learning.semantics import Lane, Line, SenseFloorSign, LaneDirection

from curriculum_learning.tasks.lane_following.agents import TeacherQuinticPolynomials, TeacherPurePursuit
from curriculum_learning.tasks.lane_following.agents import LearnerOneResidualMDN
from curriculum_learning.tasks.lane_following.regimes import OnlineLaneFollowing

LANE_WIDTH = 3
ARROWS_SCALE = 8


def world_configuration(world_width, world_height):
    world = VisualWorld(width=world_width, height=world_height)
    world.set_engine(Engine(world))  # TODO: Smelly design

    lines = [
        Line(start=(0, 0.2), end=(world_width, 0.2)),
        Line(start=(0, world_height / 2), end=(world_width, world_height / 2), discontinuous=True, color='y'),
        Line(start=(0, world_height - 0.2), end=(world_width, world_height - 0.2))
    ]
    lines_visuals = [
        LinePyplot(line=lines[0], width=LANE_WIDTH),
        # LinePyplot(line=lines[1], width=LANE_WIDTH),
        LinePyplot(line=lines[2], width=LANE_WIDTH),
    ]

    lanes = [
        Lane(lines[0], lines[1], direction=LaneDirection.START_TO_END),
        Lane(lines[1], lines[2], direction=LaneDirection.END_TO_START)
    ]

    signs = [
        SenseFloorSign(lanes[0]),
        SenseFloorSign(lanes[1])
    ]
    signs_visuals = [
        DirectionSignPyplot(floor_sign=signs[0], scale=ARROWS_SCALE),
        DirectionSignPyplot(floor_sign=signs[1], scale=ARROWS_SCALE),
    ]

    world.semantics.extend(lines)
    world.visuals.extend(lines_visuals)

    world.semantics.extend(lanes)

    world.semantics.extend(signs)
    world.visuals.extend(signs_visuals)

    return world, lanes


def world_configuration2():
    pass


LANE = 1


def online_learning(world_height, world_width):
    world, lanes = world_configuration(world_width=world_width, world_height=world_height)

    agents = [
        LearnerOneResidualMDN(world=world, x=10, y=world_height/4, theta=0.0, v=1.0),
        TeacherPurePursuit(world=world, lane=lanes[0])  # lane=lanes[0], x=0, y=32, theta=0.0, v=2
    ]
    agents_visuals = [
        PacmanAvatarPyplot(world=world, radius=world_height / 8, color=(1.0, 0.83, 0)),  # the learner controls
    ]

    world.semantics.extend(agents)
    world.visuals.extend(agents_visuals)

    learning_regime = OnlineLaneFollowing(lanes, world, agents[1], agents[0])
    learning_regime.demo_learner()


