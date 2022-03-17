import imp
from real_robots.policy import BasePolicy
import matplotlib.pyplot as plt
import numpy as np
import cv2
import utils
import torch
from skimage import io
import yaml
from config import config
import agent
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
        self.agent = agent.Agent()

        self.latent_goal = self.agent.bvae.sample_latent()

        torch.autograd.set_detect_anomaly(True)
        self.loss = 0
        self.time_step = 0
        self.internal_time_step = 0

        self.previous_observation = None
        self.action = None
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
    def internal_step(self,observation):
        if (self.internal_time_step) % config['goal_generation_freq'] == 0:
            self.latent_goal = self.agent.bvae.sample_latent()

        self.action = self.action_space.sample() #remove when we get action out of agent
      
        if self.internal_time_step < config['start_timesteps']:
            self.action = self.action_space.sample()

        else:
            if self.internal_time_step==config['start_timesteps']:
                for _ in range(config['BVAE_pretrain_steps']):
                    self.agent.train_BVAE(self.buffer)


            if (self.internal_time_step+1)%config['training_BVAE_freq']==0:
                self.agent.train_BVAE(self.buffer)


            if (self.internal_time_step+1)%config['training_agent_freq']==0:
                self.agent.train(self.buffer)

        self.internal_time_step +=1
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


        



        # self.loss = 0
        # if (self.time_step+1)%100==0 and self.time_step > config['start_timesteps']:
        #     g = []
        #     for i in range(10):
        #         generated = self.agent.bvae.sample()
        #         g.append(generated.detach().cpu().squeeze().numpy().transpose(1,2,0)[:,:,[2,1,0]])
            
        #     cv2.imshow('img',np.concatenate(g,axis=1))
            
        #     cv2.waitKey(1)
        if (self.time_step+1) %config['internal_step_freq']==0:
            self.render = True


        if self.time_step %config['internal_step_freq']==0:
            if self.previous_observation is not None:   
                self.buffer.add(self.previous_observation,self.action,observation,self.latent_goal)
    
            self.internal_step(observation)
            self.previous_observation = observation
            self.render = False


        self.time_step+=1
        
        return self.action

SubmittedPolicy=RandomPolicy
