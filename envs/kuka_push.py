# Created by giuseppe
# Date: 22/11/19

import os, random, time, math, inspect
import gym
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_envs, pybullet_data
from pybullet_envs.bullet import kukaGymEnv
import gym_kuka.kuka_no_tray as kuka
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
env_dir = os.path.abspath(os.path.join(currentdir, os.pardir))

largeValObservation = 100

RENDER_HEIGHT = 400
RENDER_WIDTH = 600
MAX_ENV_STEPS = 2000

# ---------------------------------------------------------------------------
class KukaPush(kukaGymEnv.KukaGymEnv):
  """
  This environment defines a Kuka arm that has to push a small cube on a table
  """
  # ---------------------------------
  def __init__(self, joint_control=True):
    """
    Constructor
    """
    self.joint_control = joint_control
    self.init_box_pose = [0.7, 0]
    super(KukaPush, self).__init__(renders=False)

    if self._kuka.joint_control:
      action_dim = 7 # The 7 joints positions
    else:
      action_dim = 4 # [dx, dy, dz, da, action_len]

    self._action_bound = 1
    action_high = np.array([self._action_bound] * action_dim)
    self.action_space = spaces.Box(-action_high, action_high)


    observationDim = len(self.get_observation())
    observation_high = np.array([largeValObservation] * observationDim)
    self.observation_space = spaces.Box(-observation_high, observation_high)
  # ---------------------------------

  # ---------------------------------
  def set_init_box_pose(self, init_box_pose):
    """
    This function is used to modify the initial pose of the box. Call it before the reset
    :param init_box_pose: [x, y] pose of the box
    :return:
    """
    self.init_box_pose = init_box_pose
  # ---------------------------------

  # ---------------------------------
  def reset(self):
    """
    Reset the environment
    """
    self.terminated = 0
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(self._timeStep)
    p.loadURDF(os.path.join(env_dir, "assets/plane.urdf"), [0, 0, -1])

    p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, 0.00000, -.630000,
               0.000000, 0.000000, 0.0, 1.0)

    xpos = self.init_box_pose[0]
    ypos = self.init_box_pose[1]
    ang = 3.14
    orn = p.getQuaternionFromEuler([0, 0, ang])
    self.blockUid = p.loadURDF(os.path.join(env_dir, "assets/cube_small.urdf"), xpos, ypos, 0.05,
                               orn[0], orn[1], orn[2], orn[3])

    p.setGravity(0, 0, -10)
    self._kuka = kuka.KukaNoTray(urdfRootPath=self._urdfRoot, timeStep=self._timeStep, joint_ctrl=self.joint_control)
    self._envStepCounter = 0
    p.stepSimulation()
    self._observation = self.get_observation()
    self.init_gripper_pose = self._observation[:3]
    return np.array(self._observation)
  # ---------------------------------

  # ---------------------------------
  def step(self, action):
    """
    Step the environment
    :param action: Action commands to be applied. In the form of [pos, action_len]
    :return: observation, 0, done, {}
    """
    pos = action
    # end = action[1]

    self._actionRepeat = 1
    for i in range(self._actionRepeat):
      self._kuka.applyAction(pos)
      p.stepSimulation()
      self._envStepCounter += 1
      if self._termination(1):
        break
    if self._renders:
      time.sleep(self._timeStep)
    self._observation = self.get_observation()

    done = self._termination(1)
    if done:
      self.reset_arm_pose()

    return np.array(self._observation), 0, done, {}
  # ---------------------------------

  # ---------------------------------
  def get_observation(self):
    """
    Calculate observation
    :return: observation in the form of: [x_ef, y_ef, z_ef, x_bl, y_bl, z_bl]. ef is the end effector, bl the block
    """
    self._observation = []

    gripperState = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaGripperIndex)
    self._observation.extend(list(gripperState[0])) # [x, y, z] of gripper

    blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
    self._observation.extend(list(blockPos)) # [x, y, z] of block
    return self._observation # [x_ef, y_ef, z_ef, x_bl, y_bl, z_bl]
  # ---------------------------------

  # ---------------------------------
  def _termination(self, end):
    """
    Check termination conditions
    :param action_len: Maximum duration of the policy actions
    :return: True if termination conditions are met, false otherwise
    """
    self._observation = self.get_observation()
    if self._observation[3] > 1.1 or self._observation[3] < 0.1:  # x position of bloc
      # print('Terminating cause of X')
      return True
    if np.abs(self._observation[4]) > 0.45:  # y position of bloc
      # print('Terminating cause of Y')
      return True
    if self._envStepCounter == MAX_ENV_STEPS -1:
      # print('Terminating cause of Max steps')
      return True
    if self._envStepCounter >= end*MAX_ENV_STEPS:
      # print('Terminating cause of Action len: {}'.format(end))
      return True
    return False
  # ---------------------------------

  # ---------------------------------
  def reset_arm_pose(self):
    """
    Function to reset the arm pose once the termination condition has been met
    :return: True if the pose has been reset, false otherwise
    """
    jointPositions = [
      1.006418, 0., -0.011401, -1.589317, 0.005379, 1.137684, -0.006539, 0.000048,
      -0.299912, 0.000000, -0.000043, 0.299960, 0.000000, -0.000200
    ]

    for jointIndex in range(self._kuka.numJoints):
      p.resetJointState(self._kuka.kukaUid, jointIndex, jointPositions[jointIndex])
  # ---------------------------------

  # ---------------------------------
  def render(self, mode="rgb_array", close=False):
    """
    Render function
    :param mode:
    :param close:
    :return:
    """
    if mode != "rgb_array":
      return np.array([])

    cam_pos = [0.65, 0.236, 0.0]
    # distance = 1.2 # For fancy view
    # pitch = 230    # For fancy view
    distance = 1.2
    pitch = 230
    yaw = -56
    roll = 0
    upAxisIndex = 2
    viewMat = p.computeViewMatrixFromYawPitchRoll(cam_pos, distance, pitch, yaw, roll, upAxisIndex)
    projMatrix = [
      0.75, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0,
      -0.02000020071864128, 0.0
    ]
    img = p.getCameraImage(width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=viewMat, projectionMatrix=projMatrix,
                           renderer=self._p.ER_BULLET_HARDWARE_OPENGL)

    rgb_array = np.array(img[2], dtype=np.uint8)
    rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))

    rgb_array = rgb_array[:, :, :3]
    return rgb_array
  # ---------------------------------
