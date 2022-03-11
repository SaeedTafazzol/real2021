import imp
from real_robots.policy import BasePolicy
import matplotlib.pyplot as plt
import numpy as np
import cv2
import utils
import torch
from skimage import io
import yaml
import BVAE


class RandomPolicy(BasePolicy):
    def __init__(self, action_space, observation_space):

        with open('options.yml', 'r') as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
            self.observation_mode = cfg['ENVIRONMENT_TYPE']
            self.action_mode = cfg['EVALUATION_ACTION_TYPE']
            self.object_num = cfg['EVALUATION_N_OBJECTS']

        self.action_space = action_space
        self.observation_space = observation_space

        self.render = True
        self.buffer = utils.ReplayBuffer(observation_space,action_space,self.observation_mode,self.action_mode)
        self.train_loader = torch.utils.data.DataLoader(self.buffer,batch_size=64,num_workers= 8, shuffle=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.auto = BVAE.autoencoder(self.device)
        self.loss = 0
        self.time_step = 0
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
        
        
        action = self.action_space.sample()
        self.buffer.add(observation,action,observation,0)
        action['render'] = self.render
        
        if (self.time_step+1)%10==0 and self.time_step>=1000:
            self.loss+= self.auto.train(self.train_loader)

        if (self.time_step+1)%100==0:
            print(self.time_step,self.loss/100)
            generated = self.auto.sample()
            cv2.imshow('gen',generated.detach().cpu().squeeze().numpy().transpose(1,2,0))
            cv2.waitKey(1)
            self.loss = 0
        self.time_step+=1
        return action

SubmittedPolicy=RandomPolicy
