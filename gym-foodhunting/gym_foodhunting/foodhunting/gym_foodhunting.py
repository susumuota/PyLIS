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
from gym.utils import seeding
import pybullet as p
import pybullet_data

from gym_foodhunting.foodhunting.robot import Robot, R2D2, R2D2Simple, R2D2Discrete, HSR, HSRSimple, HSRDiscrete


class FoodHuntingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    GRAVITY = -10.0
    BULLET_STEPS = 120 # p.setTimeStep(1.0 / 240.0), so 1 gym step == 0.5 sec.

    def __init__(self, render=False, robot_model=R2D2, max_steps=500, num_foods=3, num_fakes=0, object_size=1.0, object_radius_scale=1.0, object_radius_offset=1.0, object_angle_scale=1.0):
        """Initialize environment.
        """
        ### gym variables
        self.observation_space = robot_model.getObservationSpace() # classmethod
        self.action_space = robot_model.getActionSpace() # classmethod
        self.reward_range = (-1.0, 1.0)
        self.seed()
        ### pybullet settings
        self.physicsClient = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        ### env variables
        self.robot_model = robot_model
        self.max_steps = max_steps
        self.num_foods = num_foods
        self.num_fakes = num_fakes
        self.object_size = object_size
        self.object_radius_scale = object_radius_scale
        self.object_radius_offset = object_radius_offset
        self.object_angle_scale = object_angle_scale
        self.plane_id = None
        self.robot = None
        self.object_ids = []
        ### episode variables
        self.steps = 0
        self.episode_rewards = 0.0

    def close(self):
        """Close environment.
        """
        p.disconnect(self.physicsClient)

    def reset(self):
        """Reset environment.
        """
        self.steps = 0
        self.episode_rewards = 0
        p.resetSimulation()
        # p.setTimeStep(1.0 / 240.0)
        p.setGravity(0, 0, self.GRAVITY)
        self.plane_id = p.loadURDF('plane.urdf')
        self.robot = self.robot_model()
        self.object_ids = []
        for i, (pos, orn) in enumerate(self._generateObjectPositions(num=(self.num_foods+self.num_fakes), radius_scale=self.object_radius_scale, radius_offset=self.object_radius_offset, angle_scale=self.object_angle_scale)):
            if i < self.num_foods:
                urdfPath = 'food_sphere.urdf'
            else:
                urdfPath = 'food_cube.urdf'
            object_id = p.loadURDF(urdfPath, pos, orn, globalScaling=self.object_size)
            self.object_ids.append(object_id)
        for i in range(self.BULLET_STEPS):
            p.stepSimulation()
        obs = self._getObservation()
        #self.robot.printAllJointInfo()
        return obs

    def step(self, action):
        """Apply action to environment, then return observation and reward.
        """
        self.steps += 1
        self.robot.setAction(action)
        reward = -1.0 * float(self.num_foods) / float(self.max_steps) # so agent needs to eat foods quickly
        for i in range(self.BULLET_STEPS):
            p.stepSimulation()
            reward += self._getReward()
        self.episode_rewards += reward
        obs = self._getObservation()
        done = self._isDone()
        pos, orn = self.robot.getPositionAndOrientation()
        info = { 'steps': self.steps, 'pos': pos, 'orn': orn }
        if done:
            info['episode'] = { 'r': self.episode_rewards, 'l': self.steps }
            # print(self.episode_rewards, self.steps)
        #print(self.robot.getBaseRollPosition(), self.robot.getTorsoLiftPosition(), self.robot.getHeadPosition(), self.robot.getArmPosition(), self.robot.getWristPosition(), self.robot.getGripperPosition()) # for HSR debug
        #print(self.robot.getHeadPosition(), self.robot.getGripperPosition()) # for R2D2 debug
        return obs, reward, done, info

    def render(self, mode='human', close=False):
        """This is a dummy function. This environment cannot control rendering timing.
        """
        pass

    def seed(self, seed=None):
        """Set random seed.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _getReward(self):
        """Detect contact points and return reward.
        """
        reward = 0
        contacted_object_ids = [ object_id for object_id in self.object_ids if self.robot.isContact(object_id) ]
        for object_id in contacted_object_ids:
            reward += 1 if self._isFood(object_id) else -1
            p.removeBody(object_id)
            self.object_ids.remove(object_id)
        return reward

    def _getObservation(self):
        """Get observation.
        """
        obs = self.robot.getObservation()
        return obs

    def _isFood(self, object_id):
        """Check if object_id is a food.
        """
        baseLink, urdfPath = p.getBodyInfo(object_id)
        return urdfPath == b'food_sphere.urdf' # otherwise, fake

    def _isDone(self):
        """Check if episode is done.
        """
        available_object_ids = [ object_id for object_id in self.object_ids if self._isFood(object_id) ]
        return self.steps >= self.max_steps or len(available_object_ids) <= 0

    def _generateObjectPositions(self, num=1, retry=100, radius_scale=1.0, radius_offset=1.0, angle_scale=1.0, angle_offset=0.5*np.pi, z=1.5, near_distance=0.5):
        """Generate food positions randomly.
        """
        def genPos():
            r = radius_scale * self.np_random.rand() + radius_offset
            a = -np.pi * angle_scale + angle_offset
            b =  np.pi * angle_scale + angle_offset
            ang = (b - a) * self.np_random.rand() + a
            return np.array([r * np.sin(ang), r * np.cos(ang), z])
        def isNear(pos, poss):
            for p, o in poss:
                if np.linalg.norm(p - pos) < near_distance:
                    return True
            return False
        def genPosRetry(poss):
            for i in range(retry):
                pos = genPos()
                if not isNear(pos, poss):
                    return pos
            return genPos()
        poss = []
        for i in range(num):
            pos = genPosRetry(poss)
            orn = p.getQuaternionFromEuler([0.0, 0.0, 2.0*np.pi*self.np_random.rand()])
            poss.append((pos, orn))
        return poss
