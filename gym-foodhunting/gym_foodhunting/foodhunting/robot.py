# -*- coding: utf-8 -*-

# Copyright 2019 Susumu OTA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import gym
import pybullet as p

class Robot:
    # default values are for R2D2 model
    URDF_PATH = 'r2d2.urdf'

    # projectionMatrix settings
    CAMERA_PIXEL_WIDTH = 64 # 64 is minimum for stable-baselines
    CAMERA_PIXEL_HEIGHT = 64 # 64 is minimum for stable-baselines
    CAMERA_FOV = 90.0
    CAMERA_NEAR_PLANE = 0.01
    CAMERA_FAR_PLANE = 100.0

    # viewMatrix settings
    CAMERA_JOINT_INDEX = 14
    CAMERA_EYE_INDEX = 1
    CAMERA_UP_INDEX = 2
    CAMERA_EYE_SCALE = 0.05
    CAMERA_TARGET_SCALE = 1.0
    CAMERA_UP_SCALE = 1.0

    # for debug
    JOINT_TYPE_NAMES = ['JOINT_REVOLUTE', 'JOINT_PRISMATIC', 'JOINT_SPHERICAL', 'JOINT_PLANAR', 'JOINT_FIXED']

    def __init__(self, urdfPath=URDF_PATH, position=[0.0, 0.0, 1.0], orientation=[0.0, 0.0, 0.0, 1.0]):
        """Make a robot model.
        """
        self.urdfPath = urdfPath
        self.robotId = p.loadURDF(urdfPath, basePosition=position, baseOrientation=orientation)
        self.projectionMatrix = p.computeProjectionMatrixFOV(self.CAMERA_FOV, float(self.CAMERA_PIXEL_WIDTH)/float(self.CAMERA_PIXEL_HEIGHT), self.CAMERA_NEAR_PLANE, self.CAMERA_FAR_PLANE);
        self.isDebugLine = False

    @classmethod
    def getObservationSpace(cls):
        """Return observation_space for gym Env class.
        """
        return gym.spaces.Box(low=0.0, high=1.0, shape=(Robot.CAMERA_PIXEL_HEIGHT, Robot.CAMERA_PIXEL_WIDTH, 4), dtype=np.float32)

    @classmethod
    def getActionSpace(cls):
        """Return action_space for gym Env class.
        """
        raise NotImplementedError

    def setAction(self, action):
        """Set action.
        """
        raise NotImplementedError

    def scaleJointVelocity(self, jointIndex, value):
        """Scale joint velocity from [-1.0, 1.0] to [-maxVelocity, maxVelocity].
        """
        # value should be from -1.0 to 1.0
        info = p.getJointInfo(self.robotId, jointIndex)
        maxVelocity = abs(info[11])
        value *= maxVelocity
        value = -maxVelocity if value < -maxVelocity else value
        value = maxVelocity if value > maxVelocity else value
        return value

    def setJointVelocity(self, jointIndex, value, scale=1.0):
        """Set joint velocity.
        """
        # value should be from -1.0 to 1.0
        value = self.scaleJointVelocity(jointIndex, value)
        value *= scale
        p.setJointMotorControl2(bodyIndex=self.robotId,
                                jointIndex=jointIndex,
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=value)

    def scaleJointPosition(self, jointIndex, value):
        """Scale joint position from [-1.0, 1.0] to [lowerLimit, upperLimit].
        """
        # value should be from -1.0 to 1.0
        info = p.getJointInfo(self.robotId, jointIndex)
        lowerLimit = info[8]
        upperLimit = info[9]
        maxVelocity = abs(info[11])
        if lowerLimit > upperLimit:
            lowerLimit, upperLimit = upperLimit, lowerLimit # swap
        # value *= max(abs(lowerLimit), abs(upperLimit)) # TODO: is it OK?
        # y - l = a (x - -1) = a (x + 1)
        # a = (u - l) / (1 - -1) = (u - l) / 2
        # y - l = (u - l) (x + 1) / 2
        # y = (u - l) (x + 1) * 0.5 + l
        value = (upperLimit - lowerLimit) * (value + 1.0) * 0.5 + lowerLimit
        value = lowerLimit if value < lowerLimit else value
        value = upperLimit if value > upperLimit else value
        return value, maxVelocity

    def setJointPosition(self, jointIndex, value):
        """Set joint position.
        """
        # value should be from -1.0 to 1.0
        value, maxVelocity = self.scaleJointPosition(jointIndex, value)
        p.setJointMotorControl2(bodyIndex=self.robotId,
                                jointIndex=jointIndex,
                                controlMode=p.POSITION_CONTROL,
                                maxVelocity=maxVelocity,
                                targetPosition=value)

    def invScaleJointPosition(self, jointIndex, value):
        """Return inverse joint position.
        """
        info = p.getJointInfo(self.robotId, jointIndex)
        lowerLimit = info[8]
        upperLimit = info[9]
        # y - -1 = a (x - l)
        # a = (1 - -1) / (u - l) = 2 / (u - l)
        # y - -1 = 2 (x - l) / (u - l)
        # y = 2 (x - l) / (u - l) - 1
        value = 2.0 * (value - lowerLimit) / (upperLimit - lowerLimit) - 1.0
        # value should be from -1.0 to 1.0
        value = -1.0 if value < -1.0 else value
        value =  1.0 if value >  1.0 else value
        return value

    def getJointPosition(self, jointIndex):
        """Return joint position.
        """
        jointPosition, jointVelocity, jointReactionForces, appliedJointMotorTorque = p.getJointState(self.robotId, jointIndex)
        return self.invScaleJointPosition(jointIndex, jointPosition)

    def scaleJointForce(self, jointIndex, value):
        """Scale joint force from [-1.0, 1.0] to [-maxForce, maxForce].
        """
        # value should be from -1.0 to 1.0
        info = p.getJointInfo(self.robotId, jointIndex)
        maxForce = abs(info[10])
        value *= maxForce
        value = -maxForce if value < -maxForce else value
        value = maxForce if value > maxForce else value
        return value

    def setJointForce(self, jointIndex, value):
        """Set joint force.
        """
        # value should be from -1.0 to 1.0
        value = self.scaleJointForce(jointIndex, value)
        p.setJointMotorControl2(bodyIndex=self.robotId,
                                jointIndex=jointIndex,
                                controlMode=p.TORQUE_CONTROL,
                                force=value)

    def getPositionAndOrientation(self):
        """Return body's position and orientation.
        """
        return p.getBasePositionAndOrientation(self.robotId)

    def isContact(self, bodyId):
        """Return True if robot contacted with other objects.
        """
        cps = p.getContactPoints(bodyA=self.robotId, bodyB=bodyId)
        return cps and len(cps) > 0

    def getContactBodyIds(self):
        """Return bodyIds that robot contacted with.
        """
        cps = p.getContactPoints(bodyA=self.robotId)
        return cps and list(set([ cp[2] for cp in cps ])) # cp[2] == bodyUniqueIdB

    def getCameraImage(self):
        """Return camera image from CAMERA_JOINT_INDEX.
        """
        # compute eye and target position for viewMatrix
        pos, orn, _, _, _, _ = p.getLinkState(self.robotId, self.CAMERA_JOINT_INDEX)
        cameraPos = np.array(pos)
        cameraMat = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3).T
        eyePos = cameraPos + self.CAMERA_EYE_SCALE * cameraMat[self.CAMERA_EYE_INDEX]
        targetPos = cameraPos + self.CAMERA_TARGET_SCALE * cameraMat[self.CAMERA_EYE_INDEX]
        up = self.CAMERA_UP_SCALE * cameraMat[self.CAMERA_UP_INDEX]
        if self.isDebugLine:
            p.addUserDebugLine(eyePos, targetPos, lineColorRGB=[1, 0, 0], lifeTime=0.1) # red line for camera vector
            p.addUserDebugLine(eyePos, eyePos + up * 0.5, lineColorRGB=[0, 0, 1], lifeTime=0.1) # blue line for up vector
        viewMatrix = p.computeViewMatrix(eyePos, targetPos, up)
        image = p.getCameraImage(self.CAMERA_PIXEL_WIDTH, self.CAMERA_PIXEL_HEIGHT, viewMatrix, self.projectionMatrix, shadow=1, lightDirection=[1, 1, 1], renderer=p.ER_BULLET_HARDWARE_OPENGL)
        return image

    def getObservation(self):
        """Return RGB and depth images from CAMERA_JOINT_INDEX.
        """
        width, height, rgbPixels, depthPixels, segmentationMaskBuffer = self.getCameraImage()
        rgba = np.array(rgbPixels, dtype=np.float32).reshape((height, width, 4))
        depth = np.array(depthPixels, dtype=np.float32).reshape((height, width, 1))
        # seg = np.array(segmentationMaskBuffer, dtype=np.float32).reshape((height, width, 1))
        rgb = np.delete(rgba, [3], axis=2) # delete alpha channel
        rgb01 = np.clip(rgb * 0.00392156862, 0.0, 1.0) # rgb / 255.0, normalize
        obs = np.insert(rgb01, [3], np.clip(depth, 0.0, 1.0), axis=2)
        # obs = np.insert(obs, [4], seg, axis=2) # TODO: normalize
        return obs

    def printJointInfo(self, index):
        """Print joint information.
        """
        jointIndex, jointName, jointType, qIndex, uIndex, flags, jointDamping, jointFriction, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity, linkName, jointAxis, parentFramePos, parentFrameOrn, parentIndex = p.getJointInfo(self.robotId, index)
        line = [ jointName.decode('ascii'), '\n\tjointIndex\t', jointIndex, '\n\tjointName\t', jointName, '\n\tjointType\t', jointType, '\t', self.JOINT_TYPE_NAMES[jointType], '\n\tqIndex\t', qIndex, '\n\tuIndex\t', uIndex, '\n\tflags\t', flags, '\n\tjointDamping\t', jointDamping, '\n\tjointFriction\t', jointFriction, '\n\tjointLowerLimit\t', jointLowerLimit, '\n\tjointUpperLimit\t', jointUpperLimit, '\n\tjointMaxForce\t', jointMaxForce, '\n\tjointMaxVelocity\t', jointMaxVelocity, '\n\tlinkName\t', linkName, '\n\tjointAxis\t', jointAxis, '\n\tparentFramePos\t', parentFramePos, '\n\tparentFrameOrn\t', parentFrameOrn, '\n\tparentIndex\t', parentIndex, '\n' ]
        print(''.join([ str(item) for item in line ]))
        #line = [ jointIndex, jointName.decode('ascii'), self.JOINT_TYPE_NAMES[jointType] ]
        #print('\t'.join([ str(item) for item in line ]))

    def printJointInfoArray(self, indexArray):
        """Print joint informations.
        """
        for index in indexArray:
            self.printJointInfo(index)

    def printAllJointInfo(self):
        """Print all joint informations.
        """
        self.printJointInfoArray(range(p.getNumJoints(self.robotId)))


class HSR(Robot):
    URDF_PATH = 'hsrb4s.urdf'

    # viewMatrix settings
    CAMERA_JOINT_INDEX = 19
    CAMERA_EYE_INDEX = 2
    CAMERA_UP_INDEX = 1
    CAMERA_EYE_SCALE = 0.01
    CAMERA_TARGET_SCALE = 1.0
    CAMERA_UP_SCALE = -1.0

    def __init__(self, urdfPath=URDF_PATH, position=[0.0, 0.0, 0.05], orientation=[0.0, 0.0, 0.0, 1.0]):
        """Make a HSR robot model.
        """
        super(HSR, self).__init__(urdfPath, position, orientation)

    # override methods
    @classmethod
    def getActionSpace(cls):
        """Return action_space for gym Env class.
        """
        n = 13
        low = -1.0 * np.ones(n)
        high = 1.0 * np.ones(n)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def setAction(self, action):
        """Set action.
        """
        self.setWheelVelocity(action[0], action[1])
        self.setBaseRollPosition(action[2])
        self.setTorsoLiftPosition(action[3])
        self.setHeadPosition(action[4], action[5])
        self.setArmPosition(action[6], action[7], action[8])
        self.setWristPosition(action[9], action[10])
        # self.setHandPosition(action[11], action[12], action[13], action[14], action[15], action[16], action[17], action[18], action[19]) # TODO
        self.setGripperPosition(action[11], action[12])

    def isContact(self, bodyId):
        """Return True if HSR contacted with other objects.
        """
        # cps = p.getContactPoints(bodyA=self.robotId, bodyB=bodyId, linkIndexA=27) # only for wrist_roll_link
        cps = p.getContactPoints(bodyA=self.robotId, bodyB=bodyId)
        return cps and len(cps) > 0

    # HSR specific methods
    def setWheelVelocity(self, left, right):
        """Set wheel's velocity.
        """
        self.setJointVelocity(2, right, 0.25)
        self.setJointVelocity(3, left, 0.25)

    def setBaseRollPosition(self, roll):
        """Set base roll position.
        """
        self.setJointPosition(1, roll)

    def setTorsoLiftPosition(self, lift):
        """Set torso lift position.
        """
        self.setJointPosition(12, lift)

    def setHeadPosition(self, pan, tilt):
        """Set head position.
        """
        self.setJointPosition(13, pan)
        self.setJointPosition(14, tilt)

    def setArmPosition(self, lift, flex, roll):
        """Set arm position.
        """
        self.setJointPosition(23, lift)
        self.setJointPosition(24, flex)
        self.setJointPosition(25, roll)

    def setWristPosition(self, flex, roll):
        """Set wrist position.
        """
        self.setJointPosition(26, flex)
        self.setJointPosition(27, roll)

    # def setHandPosition(self, motor, leftProximal, leftSpringProximal, leftMimicDistal, leftDistal, rightProximal, rightSpringProximal, rightMimicDistal, rightDistal): # TODO
    #     """Set hand position.
    #     """
    #     self.setJointPosition(30, motor)
    #     self.setJointPosition(31, leftProximal)
    #     self.setJointPosition(32, leftSpringProximal)
    #     self.setJointPosition(33, leftMimicDistal)
    #     self.setJointPosition(34, leftDistal)
    #     self.setJointPosition(37, rightProximal)
    #     self.setJointPosition(38, rightSpringProximal)
    #     self.setJointPosition(39, rightMimicDistal)
    #     self.setJointPosition(40, rightDistal)

    def setGripperPosition(self, left, right):
        """Set gripper position.
        """
        self.setJointPosition(30, left)
        self.setJointPosition(32, right)

    def getBaseRollPosition(self):
        """Get base roll position.
        """
        return self.getJointPosition(1)

    def getTorsoLiftPosition(self):
        """Get torso lift position.
        """
        return self.getJointPosition(12)

    def getHeadPosition(self):
        """Get head position.
        """
        return self.getJointPosition(13), self.getJointPosition(14)

    def getArmPosition(self):
        """Get arm position.
        """
        return self.getJointPosition(23), self.getJointPosition(24), self.getJointPosition(25)

    def getWristPosition(self):
        """Get wrist position.
        """
        return self.getJointPosition(26), self.getJointPosition(27)

    def getGripperPosition(self):
        """Get gripper position.
        """
        return self.getJointPosition(30),  self.getJointPosition(32)


class HSRSimple(HSR):
    @classmethod
    def getActionSpace(cls):
        """Return action_space for gym Env class.
        """
        n = 2
        low = -1.0 * np.ones(n)
        high = 1.0 * np.ones(n)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def setAction(self, action):
        """Set action.
        """
        self.setWheelVelocity(action[0], action[1])
        self.setBaseRollPosition(0.0)
        self.setTorsoLiftPosition(-1.0)
        self.setHeadPosition(0.0, -0.5)
        self.setArmPosition(0.5, -1.0, 0.0)
        self.setWristPosition(0.5, 0.0)
        self.setGripperPosition(1.0, 1.0)

class HSRDiscrete(HSR):
    ACTIONS = [ [ 1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [0.0, 0.0] ]

    @classmethod
    def getActionSpace(cls):
        """Return action_space for gym Env class.
        """
        return gym.spaces.Discrete(len(cls.ACTIONS))

    def setAction(self, action):
        """Set action.
        """
        self.setWheelVelocity(*self.ACTIONS[action])
        self.setBaseRollPosition(0.0)
        self.setTorsoLiftPosition(-1.0)
        self.setHeadPosition(0.0, -0.5)
        self.setArmPosition(0.5, -1.0, 0.0)
        self.setWristPosition(0.5, 0.0)
        self.setGripperPosition(1.0, 1.0)

class R2D2(Robot):
    URDF_PATH = 'r2d2.urdf'

    # viewMatrix settings
    CAMERA_JOINT_INDEX = 14
    CAMERA_EYE_INDEX = 1
    CAMERA_UP_INDEX = 2
    CAMERA_EYE_SCALE = 0.05
    CAMERA_TARGET_SCALE = 1.0
    CAMERA_UP_SCALE = 1.0

    def __init__(self, urdfPath=URDF_PATH, position=[0.0, 0.0, 0.5], orientation=[0.0, 0.0, -1.0, 1.0]):
        """Make a R2D2 robot model.
        """
        super(R2D2, self).__init__(urdfPath, position, orientation)

    # override methods
    @classmethod
    def getActionSpace(cls):
        """Return action_space for gym Env class.
        """
        n = 6
        low = -1.0 * np.ones(n)
        high = 1.0 * np.ones(n)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def setAction(self, action):
        """Set action.
        """
        self.setWheelVelocity(action[0], action[1])
        self.setGripperPosition(action[2], action[3], action[4])
        self.setHeadPosition(action[5])

    # R2D2 specific methods
    def setWheelVelocity(self, left, right):
        """Set wheel's velocity.
        """
        self.setJointVelocity(2, right, -0.1)
        self.setJointVelocity(3, right, -0.1)
        self.setJointVelocity(6, left, -0.1)
        self.setJointVelocity(7, left, -0.1)

    def setGripperPosition(self, extension, left, right):
        """Set gripper position.
        """
        self.setJointPosition(8, extension)
        self.setJointPosition(9, left)
        self.setJointPosition(11, right)

    def setHeadPosition(self, pan):
        """Set head position.
        """
        self.setJointPosition(13, pan)

    def getGripperPosition(self):
        """Get gripper position.
        """
        return self.getJointPosition(8), self.getJointPosition(9), self.getJointPosition(11)

    def getHeadPosition(self):
        """Get head position.
        """
        return self.getJointPosition(13)

class R2D2Simple(R2D2):
    @classmethod
    def getActionSpace(cls):
        """Return action_space for gym Env class.
        """
        n = 2
        low = -1.0 * np.ones(n)
        high = 1.0 * np.ones(n)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def setAction(self, action):
        """Set action.
        """
        self.setWheelVelocity(action[0], action[1])
        self.setGripperPosition(1.0, 1.0, 1.0)
        self.setHeadPosition(0.0)

class R2D2Discrete(R2D2):
    ACTIONS = [ [ 1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [0.0, 0.0] ]

    @classmethod
    def getActionSpace(cls):
        """Return action_space for gym Env class.
        """
        return gym.spaces.Discrete(len(cls.ACTIONS))

    def setAction(self, action):
        """Set action.
        """
        self.setWheelVelocity(*self.ACTIONS[action])
        self.setGripperPosition(1.0, 1.0, 1.0)
        self.setHeadPosition(0.0)