# def episodic(self):
#     for episode in self.episodes:
#         for step in episode:
#             self.teacher.execute(step)
#             self.learner.exploit(self.world.observe(), horizon=10)
#             self.learner.execute()
#
# def generate_episodes(self):
#     self.episodes = []
#
#     starting_points = self.lane_space_sampling()
#     for starting_point in starting_points:
#         self.teacher.x = starting_point[0]
#         self.teacher.y = starting_point[1]
#         self.teacher.theta = starting_point[2]
#         self.teacher.v = 1.0
#         plan = self.teacher.plan()
#         self.episodes.append(plan)
#
#     # for x, y, theta in starting_points:
#     #     plt.arrow(x, y, 5.0 * math.cos(theta), 5.0 * math.sin(theta),
#     #               fc='r', ec='k', head_width=2.5, head_length=2.5)
#     #
#     # plt.pause(10)
