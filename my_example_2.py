"""
An example of how one might use PyRep to create their RL environments.
In this case, the Franka Panda must reach a randomly placed target.
This script contains examples of:
    - RL environment example.
    - Scene manipulation.
    - Environment resets.
    - Setting joint properties (control loop disabled, motor locked at 0 vel)
"""
from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.robots.mobiles.turtlebot import TurtleBot
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
import numpy as np

SCENE_FILE = join(dirname(abspath(__file__)),
                  'scene_turtlebot_navigation.ttt')
POS_MIN, POS_MAX = [0, 1, 0.1], [0, 1, 0.1]
EPISODES = 10
EPISODE_LENGTH = 500


class ReacherEnv(object):

    def __init__(self):
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=False)
        self.pr.start()
        self.agent = TurtleBot()
        self.agent.set_control_loop_enabled(False)
        self.agent.set_motor_locked_at_zero_velocity(True)
        self.target = Shape.create(type=PrimitiveShape.SPHERE,
                              size=[0.05, 0.05, 0.05],
                              color=[7.0, 0.5, 0.5],
                              static=True, respondable=False)
        # self.target = Shape('target')
        # self.agent_ee_tip = self.agent.get_tip()
        self.initial_joint_positions = self.agent.get_joint_positions()

    def _get_state(self):
        # Return state containing arm joint angles/velocities & target position
        return np.concatenate([self.agent.get_base_actuation(),
                               self.agent.get_base_velocities(),
                               self.target.get_position()])

    def reset(self):
        # Get a random position within a cuboid and set the target position
        pos = list(np.random.uniform(POS_MIN, POS_MAX))
        pos_agent = list(np.random.uniform([0, 0, 0], [0, 0, 0]))
        self.target.set_position(pos)
        self.agent.set_position(pos_agent)
        return self._get_state()

    def step(self, action):
        self.agent.set_joint_target_velocities(action)  # Execute action on arm
        self.pr.step()  # Step the physics simulation
        ax, ay, az = self.agent.get_position()
        tx, ty, tz = self.target.get_position()
        # Reward is negative distance to target
        reward = -np.sqrt((ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2)
        print(int(reward*100),int(ax*100),int(ay*100),int(az*100),int(tx*100),int(ty*100),int(tz*100))
        return reward, self._get_state()

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()


class Agent(object):

    def act(self, state):
        del state
        return list(np.random.uniform(-30.0, 30.0, size=(2,)))

    def learn(self, replay_buffer):
        del replay_buffer
        pass


env = ReacherEnv()
agent = Agent()
replay_buffer = []

for e in range(EPISODES):

    print('Starting episode %d' % e)
    state = env.reset()
    for i in range(EPISODE_LENGTH):
        action = agent.act(state)
        reward, next_state = env.step(action)
        replay_buffer.append((state, action, reward, next_state))
        state = next_state
        agent.learn(replay_buffer)

print('Done!')
env.shutdown()
