import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class Representation(nn.Module):
	def __init__(self):
		super(Representation, self).__init__()

		self.conv1 = nn.Conv2d(config['input_channels'], config['conv1'][0], config['conv1'][1],
							   stride=config['conv1'][2])
		self.conv2 = nn.Conv2d(config['conv1'][0], config['conv2'][0], config['conv2'][1], stride=config['conv2'][2])
		self.conv3 = nn.Conv2d(config['conv2'][0], config['conv3'][0], config['conv3'][1], stride=config['conv3'][2])

	def forward(self, input):
		out = F.leaky_relu(self.conv1(input))
		out = F.leaky_relu(self.conv2(out))
		out = F.leaky_relu(self.conv3(out))

		out = out.reshape(-1, np.prod(config['output_conv3']))
		return out


class Actor(nn.Module):
	def __init__(self):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(np.prod(config['output_conv3']), config['agent_hidden'])
		self.l2 = nn.Linear(config['agent_hidden'], config['agent_hidden'])
		self.l3 = nn.Linear(config['agent_hidden'], config['action_dim'])

		self.max_action = torch.tensor(self.config['max_action']).to(config['device'])
		self.min_action = torch.tensor(self.config['min_action']).to(config['device'])

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))

		# self.max_action * torch.tanh(self.l3(a))
		a = (self.max_action - self.min_action) * torch.sigmoid(self.l3(a)) + self.min_action
		return a


class Critic(nn.Module):
	def __init__(self):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(np.prod(config['output_conv3']) + config['action_dim'], config['agent_hidden'])
		self.l2 = nn.Linear(config['agent_hidden'], config['agent_hidden'])
		self.l3 = nn.Linear(config['agent_hidden'], 1)

		# Q2 architecture
		self.l4 = nn.Linear(np.prod(config['output_conv3']) + config['action_dim'], config['agent_hidden'])
		self.l5 = nn.Linear(config['agent_hidden'], config['agent_hidden'])
		self.l6 = nn.Linear(config['agent_hidden'], 1)

	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2

	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class Agent(object):
	def __init__(
			self,
			discount=0.99,
			tau=0.005,
			policy_noise=0.2,
			noise_clip=0.5,
			policy_freq=2
	):
		self.representation = Representation()
		self.representation_optimizer = torch.optim.Adam(self.representation.parameters(), lr=3e-4)

		self.actor = Actor().to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic().to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0


def select_action(self, state):
	state = torch.FloatTensor(state.reshape(1, -1)).to(device)
	return self.actor(state).cpu().data.numpy().flatten()


def train(self, replay_buffer, batch_size=256):
	self.total_it += 1

	# Sample replay buffer
	state_raw, action, next_state_raw, reward, not_done = replay_buffer.sample(batch_size)

	# Get state, next_state representations
	state = self.representation(state_raw)
	next_state = self.representation(next_state_raw)

	with torch.no_grad():
		# Select action according to policy and add clipped noise
		noise = (
				torch.randn_like(action) * self.policy_noise
		).clamp(-self.noise_clip, self.noise_clip)

		next_action = (
				self.actor_target(next_state) + noise
		).clamp(self.min_action, self.max_action)

		# Compute the target Q value
		target_Q1, target_Q2 = self.critic_target(next_state, next_action)
		target_Q = torch.min(target_Q1, target_Q2)
		target_Q = reward + not_done * self.discount * target_Q

	# Get current Q estimates
	current_Q1, current_Q2 = self.critic(state, action)

	# Compute critic loss
	critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

	# Optimize the critic
	self.critic_optimizer.zero_grad()
	self.representation_optimizer.zero_grad()
	critic_loss.backward()
	self.critic_optimizer.step()
	# Optimise the representation
	self.representation_optimizer.step()

	# Delayed policy updates
	if self.total_it % self.policy_freq == 0:

		# Compute actor losse
		actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

		# Optimize the actor
		self.actor_optimizer.zero_grad()
		self.representation_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()
		# Optimise the representation
		self.representation_optimizer.step()

		# Update the frozen target models
		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


def save(self, filename):
	torch.save(self.critic.state_dict(), filename + "_critic")
	torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

	torch.save(self.actor.state_dict(), filename + "_actor")
	torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

	torch.save(self.representation.state_dict(), filename + "_representation")
	torch.save(self._representation_optimizer.state_dict(), filename + "__representation_optimizer")

def load(self, filename):
	self.critic.load_state_dict(torch.load(filename + "_critic"))
	self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
	self.critic_target = copy.deepcopy(self.critic)

	self.actor.load_state_dict(torch.load(filename + "_actor"))
	self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
	self.actor_target = copy.deepcopy(self.actor)

	self._representation.load_state_dict(torch.load(filename + "__representation"))
	self._representation_optimizer.load_state_dict(torch.load(filename + "__representation_optimizer"))
