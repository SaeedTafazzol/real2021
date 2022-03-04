import imp
from real_robots.policy import BasePolicy
import matplotlib.pyplot as plt
import numpy as np
import cv2
import utils
import torch
from skimage import io
class RandomPolicy(BasePolicy):
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space

        self.render = True
        print(observation_space['retina'].shape)

        self.buffer = utils.ReplayBuffer(np.array(observation_space['retina'].shape)[[2,0,1]],1)
        self.ctr = 0

    def start_intrinsic_phase(self):
        """
        The evaluator will call this function to signal the start of the
        Intrinsic Phase.
        Next observation will be from the intrinsic phase.
        """
        pass

    def end_intrinsic_phase(self, observation, reward, done):
        """
        The evaluator will call this function to signal the end of the
        Intrinsic Phase.
        It passes the observation, reward and done flag values computed at
        the end of the last step of the Intrinsic Phase.
        """
        pass

    def start_extrinsic_phase(self):
        """
        The evaluator will call this function to signal the start of the
        Extrinsic Phase.
        Next observation will be from the extrinsic phase.
        """
        pass

    def end_extrinsic_phase(self):
        """
        The evaluator will call this function to signal the end of the
        Extrinsic Phase.
        """
        pass

    def start_extrinsic_trial(self):
        """
        The evaluator will call this function to signal the start of each
        extrinsic trial.
        Next observation will have a (new) goal.
        """
        pass

    def end_extrinsic_trial(self, observation, reward, done):
        """
        The evaluator will call this function to signal the end of each
        extrinsic trial.
        It passes the observation, reward and done flag values computed at
        the end of the last step of the extrinsic trial.
        """
        pass

    def step(self, observation, reward, done):
        """
        The step function will receive the observation, reward and done signals
        from the environment and computes the next action to take.
        Parameters
        ----------
        observation : dict
            The dictionary will contain the following entries:
                - "joint_positions"
                    Values of the joints position of the arm,
                    including the gripper.
                - "touch_sensors"
                    Values recorded by the touch sensors.
                - "retina"
                    Image of the environment from the camera
                    above the table.
                - "goal"
                    Image of the goal, showing how the objects
                    should be arranged in the environment.
            If the ENVIRONMENT_TYPE is "easy", these additional
            observations are also provided in the same dictionary:
                - "object_positions"
                    a dictionary with a key for each object on the table with
                    associated position and orientation of the object
                - "goal_positions"
                    a dictionary with the goal position of each object
                - "mask"
                    a segmentation mask of the retina image where for each
                    pixel there is an integer index that identifies which
                    object is in that pixel (i.e. -1 is a background pixel,
                    0 is the robot, 1 is the table, etc).
                - "goal_mask"
                    a segmentation mask of the goal image
        reward: float
            This will be always zero.
        done: bool
            This will be True when:
                - intrisic phase ends
                - an extrinsic trial ends
            otherwise it will always be false.
        """
        # print(observation['retina'].shape)
        # print(observation['retina'])
        # cv2.imshow('retina', observation['retina'])
        # cv2.waitKey(1) 
        
        self.buffer.add(torch.tensor(observation['retina'].transpose(2,0,1)))
        
        action = self.action_space.sample()
        io.imsave('./img/'+str(self.ctr)+'.png',observation['retina'])
        self.ctr+=1
        # print(action)
        # print(self.action_space)
        action['render'] = self.render
        return action

SubmittedPolicy=RandomPolicy
