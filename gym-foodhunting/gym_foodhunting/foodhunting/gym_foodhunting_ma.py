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

class FoodHuntingMAEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    GRAVITY = -10.0
    BULLET_STEPS = 120 # p.setTimeStep(1.0 / 240.0), so 1 gym step == 0.5 sec.

    def __init__(self, render=False, robot_model=R2D2, max_steps=500, num_agents=1, num_foods=1):
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
        ### episode variables
        self.steps = 0
        self.episode_rewards = []
        ### env variables
        self.robot_model = robot_model
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.num_foods = num_foods
        self.robots = []
        self.food_ids = []
        self.policies = []

    def setPolicies(self, policies):
        self.policies = policies

    def close(self):
        """Close environment.
        """
        p.disconnect(self.physicsClient)

    def reset(self):
        """Reset environment.
        """
        self.steps = 0
        self.episode_rewards = [ 0.0 for _ in range(self.num_agents) ]
        p.resetSimulation()
        # p.setTimeStep(1.0 / 240.0)
        p.setGravity(0, 0, self.GRAVITY)
        p.loadURDF('plane.urdf')
        poss = self._generateObjectPositions(num=self.num_agents+self.num_foods)
        self.robots = [ self.robot_model(position=poss[i][0], orientation=poss[i][1]) for i in range(self.num_agents) ]
        self.robots[0].isDebugLine = True
        self.food_ids = [ p.loadURDF('food_sphere.urdf', poss[self.num_agents+i][0], poss[self.num_agents+i][1]) for i in range(self.num_foods) ]
        for i in range(self.BULLET_STEPS):
            p.stepSimulation()
        obs = self.robots[0].getObservation()
        return obs

    def step(self, action):
        """Apply action to environment, then return observation and reward.
        """
        self.steps += 1
        self.robots[0].setAction(action)
        for i in range(self.num_agents):
            if i != 0 and self.policies[i:i+1]: # self.policies[0] is dummy
                self.robots[i].setAction(self.policies[i](self.robots[i].getObservation()))
        # rewards = [ -1.0 * self.num_foods / self.max_steps for _ in range(self.num_agents) ] # so agent needs to eat foods quickly
        rewards = [ 0.0 for _ in range(self.num_agents) ]
        for i in range(self.BULLET_STEPS):
            p.stepSimulation()
            rewards = [ rewards[i]+self._getReward(self.robots[i]) for i in range(self.num_agents) ]
        self.episode_rewards = [ self.episode_rewards[i]+rewards[i] for i in range(self.num_agents) ]
        obs = self.robots[0].getObservation()
        done = self._isDone()
        info = { 'steps': self.steps }
        if done:
            # TODO
            info['episode'] = { 'r': self.episode_rewards[0], 'l': self.steps, 'r_all': self.episode_rewards }
            # print(self.episode_rewards, self.steps)
        return obs, rewards[0], done, info

    def render(self, mode='human', close=False):
        """This is a dummy function. This environment cannot control rendering timing.
        """
        pass

    def seed(self, seed=None):
        """Set random seed.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _getReward(self, robot):
        """Detect contact points and return reward.
        """
        reward = 0
        for food_id in set(self.food_ids) & set(robot.getContactBodyIds()):
            reward += 1
            p.removeBody(food_id)
            self.food_ids.remove(food_id)
        return reward

    def _isDone(self):
        """Check if episode is done.
        """
        return self.steps >= self.max_steps or len(self.food_ids) <= 0

    def _generateObjectPositions(self, num=1, retry=100, radius_scale=1.0, radius_offset=1.0, angle_scale=1.0, angle_offset=0.5*np.pi, z=0.5, near_distance=1.0):
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
        self.np_random.shuffle(poss)
        return poss
