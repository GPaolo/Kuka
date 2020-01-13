# Created by giuseppe
# Date: 26/11/19

import pybullet as p
import os
import pybullet_envs, pybullet_data
from pybullet_envs.bullet import kuka
import math
import numpy as np

class KukaNoTray(kuka.Kuka):
  """
  This class loads the kuka arm for the pybullet simulation, without loading the tray on the table.
  """
  # ---------------------------------
  def __init__(self, urdfRootPath=pybullet_data.getDataPath(), timeStep=0.01, joint_ctrl=True):
    super(KukaNoTray, self).__init__(urdfRootPath=urdfRootPath, timeStep=timeStep)
    self.joint_control = joint_ctrl
  # ---------------------------------

  # ---------------------------------
  def applyAction(self, motorCommand):
    jointPoses = motorCommand

    if not self.joint_control:
      state = p.getLinkState(self.kukaUid, self.kukaEndEffectorIndex)

      pos = np.append(motorCommand, 0.3)

      if (pos[0] > 1):
        pos[0] = 1
      if (pos[0] < 0):
        pos[0] = 0
      if (pos[1] < -.45):
        pos[1] = -.45
      if (pos[1] > 0.45):
        pos[1] = 0.45

      orn = p.getQuaternionFromEuler([0, -math.pi, 0])  # -math.pi,yaw])
      jointPoses = p.calculateInverseKinematics(self.kukaUid,
                                                self.kukaEndEffectorIndex,
                                                pos,
                                                orn,
                                                jointDamping=self.jd)

    for i in range(len(jointPoses)):
      # print(i)
      p.setJointMotorControl2(bodyUniqueId=self.kukaUid,
                              jointIndex=i,
                              controlMode=p.POSITION_CONTROL,
                              targetPosition=jointPoses[i],
                              force=self.maxForce,
                              maxVelocity=self.maxVelocity*3,
                              # positionGain=0.3,
                              # velocityGain=0.1
                              )

  # ---------------------------------

  def reset(self):
    objects = p.loadSDF(os.path.join(self.urdfRootPath, "kuka_iiwa/kuka_with_gripper2.sdf"))
    self.kukaUid = objects[0]
    #for i in range (p.getNumJoints(self.kukaUid)):
    #  print(p.getJointInfo(self.kukaUid,i))
    p.resetBasePositionAndOrientation(self.kukaUid, [-0.100000, 0.000000, 0.070000],
                                      [0.000000, 0.000000, 0.000000, 1.000000])
    self.jointPositions = [
        # 0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684, -0.006539, 0.000048,
        # -0.299912, 0.000000, -0.000043, 0.299960, 0.000000, -0.000200
      0.004650348419804516, 0.6956324512525387, -0.009067755447290828, -1.7286201257556935, 0.008833701273411756,
       0.7173796539426073, -0.008965729120192811, 1.7617430024151515e-05, 0, -0.1356631921973787, -2.5877220958089648e-05, 0,
       0.1378489743848122, -7.167140187666544e-05

    ]
    self.numJoints = p.getNumJoints(self.kukaUid)
    for jointIndex in range(self.numJoints):
      p.resetJointState(self.kukaUid, jointIndex, self.jointPositions[jointIndex])
      p.setJointMotorControl2(self.kukaUid,
                              jointIndex,
                              p.POSITION_CONTROL,
                              targetPosition=self.jointPositions[jointIndex],
                              force=self.maxForce)
    self.endEffectorPos = [0.537, 0.0, 0.3]
    self.endEffectorAngle = 0

    self.motorNames = []
    self.motorIndices = []

    for i in range(self.numJoints):
      jointInfo = p.getJointInfo(self.kukaUid, i)
      qIndex = jointInfo[3]
      if qIndex > -1:
        #print("motorname")
        #print(jointInfo[1])
        self.motorNames.append(str(jointInfo[1]))
        self.motorIndices.append(i)
