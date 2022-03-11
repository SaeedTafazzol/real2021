import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import *
from config import config


class Encoder(nn.Module):
  def __init__(self,z=10):
    super(Encoder, self).__init__()

    self.conv1 = nn.Conv2d(config['input_channels'],config['conv1'][0],config['conv1'][1],stride=config['conv1'][2])
    self.conv2 = nn.Conv2d(config['conv1'][0],config['conv2'][0],config['conv2'][1],stride=config['conv2'][2])
    self.conv3 = nn.Conv2d(config['conv2'][0],config['conv3'][0],config['conv3'][1],stride=config['conv3'][2])

    
    self.fc = nn.Linear(np.prod(config['output_conv3']),128)
 
    self.mean = nn.Linear(128 ,32)
    self.log_std = nn.Linear(128 ,32)

  def forward(self, img):
    out = F.leaky_relu(self.conv1(img))
    out = F.leaky_relu(self.conv2(out))
    out = F.leaky_relu(self.conv3(out))

    out = out.reshape(-1,np.prod(config['output_conv3']))
    out = F.leaky_relu(self.fc(out))
    mean = 	F.tanh(self.mean(out))
    log_std = self.log_std(out)
    return mean,log_std


class Decoder(nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()

    self.fc1 = nn.Linear(32,128)
    self.fc2 = nn.Linear(128,np.prod(config['output_conv3']))
    self.conv1 = nn.Conv2d(config['conv3'][0], config['conv2'][0],config['conv3'][1],padding='same')
    self.conv2 = nn.Conv2d(config['conv2'][0],config['conv1'][0],config['conv2'][1],padding='same')
    self.conv3 = nn.Conv2d(config['conv1'][0],config['input_channels'],config['conv1'][1],padding='same')

  def forward(self, lat):
    out = F.leaky_relu(self.fc1(lat))
    out = F.leaky_relu(self.fc2(out))
    out = out.reshape(-1,config['output_conv3'][0], config['output_conv3'][1], config['output_conv3'][2])
    out = F.leaky_relu(self.conv1(F.interpolate(out,(config['output_conv2'][1],config['output_conv2'][2]),
                                                mode='bilinear')))
    out = F.leaky_relu(self.conv2(F.interpolate(out,(config['output_conv1'][1],config['output_conv1'][2]),
                                                mode='bilinear')))
    out = F.sigmoid(self.conv3(F.interpolate(out,(config['image_size'][1],config['image_size'][2]),mode='bilinear')))

    return	out

class autoencoder(object):
  def __init__(self,device):
    self.device = device
    self.decoder = Decoder().to(self.device)
    self.encoder = Encoder().to(self.device)
    self.optimizer =  torch.optim.Adam(list(self.encoder.parameters())+list(self.decoder.parameters()),lr=1e-3)



  def train(self,data_loader):
    observation,action,next_observation,goal = next(iter(data_loader))
    batch_size = data_loader.batch_size
    img = torch.tensor(observation['retina'],dtype=torch.float)
    img = img.to(self.device)
    beta = 5
    mean,log_std = self.encoder(img)

    var = torch.exp(2*log_std)
 
    a = torch.arange(0,var.shape[1]).to(self.device)
    cov_mat = torch.zeros((var.shape[0],var.shape[1],var.shape[1])).to(self.device)
    cov_mat[:,a,a] = var

    nor_dist = torch.distributions.MultivariateNormal(mean, cov_mat)

    standard_nor = torch.distributions.MultivariateNormal(torch.zeros(mean.shape[1]).to(self.device), torch.eye(cov_mat.shape[1]).to(self.device))

    latent = standard_nor.sample((mean.shape[0],))*torch.exp(log_std) + mean

    bern = self.decoder(latent)
    
    loss =  nn.BCELoss(reduction='sum')(bern,img)/torch.tensor(batch_size) + beta*kl_divergence(nor_dist,standard_nor).mean(0)
    self.optimizer.zero_grad()


    loss.backward()

    self.optimizer.step()
    
    return loss
  def sample(self):
    standard_nor = torch.distributions.MultivariateNormal(torch.zeros(32,).to(self.device), torch.eye(32,).to(self.device))
    latent = standard_nor.sample()
    bern = self.decoder(latent)
    return bern