import numpy
import torch
import numpy as np

class CropImage(object):
    def __init__(self, mode='easy'):
        # Easy mode has a mask
        self.observation_mode = mode

    def __call__(self, observation:dict) -> numpy.ndarray:
        if self.observation_mode=='easy':
            transformed = np.concatenate((np.expand_dims(observation['mask'],-1),observation['retina']/255),axis=-1)[0:180,70:250,:]
        else:
            transformed = (observation['retina']/255)[0:180,70:250,:]
        return transformed

class ToTensor(object):
    """Convert observation dict to Tensors."""

    def __init__(self, device):
        self.device = device

    def __call__(self, observation:dict) ->dict :
        for key in observation.keys():
            if type(observation[key])!=list:# and observation[key].dtype == np.float64:
                observation[key] = torch.tensor(observation[key],dtype=torch.float).to(self.device)

        return observation