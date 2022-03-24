

from scipy import rand
from config import config
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from skimage import io
import os
import cv2
from collections import OrderedDict
import random
from RealTranforms import CropImage, ToTensor
from torchvision.transforms import Compose

def parse_action(action):
	return np.concatenate((action['cartesian_command'],action['gripper_command']))

def wrap_action(action):
		return OrderedDict({'cartesian_command':action[0:7],'gripper_command':action[7:9],'render':False})



class ReplayBuffer:
	def __init__(self,observation_space,action_space,observation_mode,action_mode):
		self.batch_size = config['batch_size']
		self.storage = Storage(observation_space,action_space,observation_mode,action_mode)
		self.train_loader = torch.utils.data.DataLoader(self.storage,batch_size=self.batch_size,num_workers= 8, shuffle=True)
		self.current_episode_idx = []

		self.composed = Compose([CropImage(observation_mode), ToTensor(config['device'])])
		self.crop = CropImage(observation_mode)
		self.toTensor = ToTensor(config['device'])
	
	def reset_episode(self):
		self.current_episode_idx = []

	def __getitem__(self, idx):
		observation,next_observation,action,goal = self.storage[idx]

		observation = self.toTensor(observation)
		next_observation = self.toTensor(next_observation)
		action = torch.tensor(action,dtype= torch.float).to(config['device'])
		goal = torch.tensor(goal,dtype= torch.float).to(config['device'])

		return observation,next_observation,action,goal


	def sample(self):
		observation,next_observation,action,goal = next(iter(self.train_loader))
		
		action = torch.tensor(action,dtype=torch.float).to(config['device'])
		goal = torch.tensor(goal,dtype=torch.float).to(config['device'])
		observation = self.toTensor(observation)
		next_observation = self.toTensor(next_observation)

		return observation,next_observation,action,goal

	def add(self, observation,action,next_observation,goal,hindsight = False):
		# Takes a full obs dict;

		# Hindsight messes up the indices so store the indices
		if not hindsight:
			self.current_episode_idx.append(self.storage.ptr)
		self.storage.add(observation,action,next_observation,goal)

		


class Storage(Dataset):
	def __init__(self, observation_space, action_space,observation_mode,action_mode, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0
		self.observation_space = observation_space
		self.action_space = action_space
		self.observation_mode = observation_mode
		self.action_mode = action_mode
		
		self.observation = np.ndarray((max_size,),object)
		self.action = np.ndarray((max_size,),object)
		self.next_observation = np.ndarray((max_size,),object)
		self.goal = np.ndarray((max_size,),object)
		
		if not os.path.exists('retina_current'):
			os.makedirs('retina_current')
		
		if not os.path.exists('retina_next'):
			os.makedirs('retina_next')

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.crop = CropImage(observation_mode)


	def add(self, observation,action,next_observation,goal):

		self.observation[self.ptr] = {
		'image_name':str(self.ptr)+'.png',
		'joint_positions':observation['joint_positions'],
		'touch_sensors':observation['touch_sensors'],
		}

		self.next_observation[self.ptr] = {
		'image_name':str(self.ptr)+'.png',
		'joint_positions':next_observation['joint_positions'],
		'touch_sensors':next_observation['touch_sensors'],
		}

		np.save('retina_current/'+str(self.ptr)+'.npy', self.crop(observation))
		np.save('retina_next/'+str(self.ptr)+'.npy', self.crop(next_observation))

		self.action[self.ptr] = parse_action(action)
		self.goal[self.ptr] = goal

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def __len__(self):
		return max(self.size,1)

	def __getitem__(self, idx):
		observation = self.observation[idx]
		next_observation = self.next_observation[idx]
		action = self.action[idx]
		goal = self.goal[idx]
		observation['retina'] = np.load('retina_current/'+str(idx)+'.npy').transpose(2,0,1)
		next_observation['retina'] = np.load('retina_next/'+str(idx)+'.npy').transpose(2,0,1)

		return observation,next_observation,action,goal

