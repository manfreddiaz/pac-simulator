
def roundabout(width=16, height=10):
    world = VisualWorld(width=width, height=height)
    world.set_engine(Engine(world))  # TODO: Smelly design

    lines_visuals = [
        CirclePyplot(center=(8, 5), radius=2.0 + LANE_WIDTH, width=LANE_THICKNESS),
        CirclePyplot(center=(8, 5), radius=2.0, width=LANE_THICKNESS)
    ]

    agents = [
        Agent(world=world, x=8 - (2 + LANE_WIDTH) / 2 - AGENT_RADIUS, y=5, theta=-pi / 4, v=0.0)
    ]
    agents_visuals = [
        PacmanAvatarPyplot(world=world, radius=AGENT_RADIUS, color=(1.0, 0.83, 0)),  # the learner controls
    ]
    world.active_agent = agents[0]
    world.visuals.extend(lines_visuals)
    world.visuals.extend(agents_visuals)

    return world