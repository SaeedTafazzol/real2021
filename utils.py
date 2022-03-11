import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from skimage import io
import os
class ReplayBuffer(Dataset):
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


		if self.observation_mode=='easy':
			np.save('retina_current/'+str(self.ptr)+'.npy',
				np.concatenate((np.expand_dims(observation['mask'],-1),observation['retina']/255),axis=-1)[0:180,70:250,:])
			np.save('retina_next/'+str(self.ptr)+'.npy',
				np.concatenate((np.expand_dims(next_observation['mask'],-1),next_observation['retina']/255),axis=-1)[0:180,70:250,:])
			self.observation[self.ptr]['object_positions'] = observation['object_positions']
			self.next_observation[self.ptr]['object_positions'] = next_observation['object_positions']
			
		else:
			np.save('retina_current/'+str(self.ptr)+'.npy',(observation['retina']/255)[0:180,70:250,:])
			np.save('retina_next/'+str(self.ptr)+'.npy',(next_observation['retina']/255)[0:180,70:250,:])


		self.action[self.ptr] = action
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
