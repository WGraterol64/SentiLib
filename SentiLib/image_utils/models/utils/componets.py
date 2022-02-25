"""
	The original EmbraceNet implementation is at https://github.com/idearibosome/embracenet
	For both WeightedSum and EmbraceNet the use of dynamic, or even trainable probabilities was also tested, but this did not provide good results but some problems.
	WeightedSum implementation is similar to EmbraceNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from itertools import combinations

class WeightedSum(nn.Module):
	def __init__(self, number_modals, outchannels, probabilities=None,
								trainable=False, mode='', device=torch.device('cpu')):
		super(WeightedSum, self).__init__()
		self.NumberModalities = number_modals
		self.Mode = mode
		self.Probabilities = probabilities
		self.Device = device
		self.set_weights(outchannels, trainable)

	def set_weights(self, outchannels, trainable):
		i = 0
		if self.Mode == 'convs':
			setattr(self, 'Weight_%d'%(i),nn.Conv1d(self.NumberModalities, outchannels, kernel_size=1, stride=1))
			if not trainable:
				getattr(self, 'Weight_%d'%(i)).weight.requires_grad=False

		elif self.Mode == 'tensor':
			if self.Probabilities is not None:
				tt = torch.tensor(self.Probabilities, dtype=torch.float)
			else:
				tt = torch.ones(self.NumberModalities, outchannels, dtype=torch.float)
			tt = torch.div(tt, torch.sum(tt, dim=0, keepdim=True)).to(self.Device)
			self.Weight_1 = nn.Parameter(tt, requires_grad=trainable)
		else:
			raise NameError('{} is not supported yet'.format(self.Mode))

	def forward(self, outs, availability):
		b,c,n = outs.shape
		W = getattr(self,'Weight_%d'%(1))
		if self.Mode == 'convs':
			out = W(outs)
		elif self.Mode == 'tensor':
			W = torch.stack([W]*b).view(b,c)
			W = torch.mul(W,availability)
			W = torch.div(W, torch.sum(W, dim=-1, keepdim=True)).view(b,c,1)
			out = torch.zeros(b, 1, n, dtype=torch.float, device=self.Device)
			for i, o in enumerate(outs):
				out[i,:,:] = (torch.mm(o.T, W[i])).T

		return out.view(out.shape[0],out.shape[-1])

class EmbraceNet(nn.Module):
	def __init__(self, input_size_list=[], embracement_size=32, docker_arch=[], device = torch.device('cpu')):
		super(EmbraceNet, self).__init__()

		self.input_size_list = input_size_list
		self.embracement_size = embracement_size
		self.bypass_docking = self.set_dockers(docker_arch)
		self.Device = device

	def set_dockers(self, docker_architecture=[]):
		'''
			return boolean for use docking or not
		'''
		bypass = True
		for i, arch in enumerate(docker_architecture):
			bypass = False
			layers = []
			inC = self.input_size_list[i]
			for l in arch:
				if l == 'D':
					layers += [nn.Dropout()]
				elif l == 'R':
					layers += [nn.ReLU()]
				else:
					layers += [nn.Linear(inC,l)]
					inC = l
			setattr(self,'docking_%d' % (i),nn.Sequential(*layers))
		return bypass 

	def forward(self, input_list, availabilities=None, selection_probabilities=None):
		assert len(input_list) == len(self.input_size_list)
		num_modalities = len(input_list)
		batch_size = input_list[0].shape[0]

		docking_output_list = []
		if self.bypass_docking:
			docking_output_list = input_list
		else:
			for i, input_data in enumerate(input_list):
				x = getattr(self, 'docking_%d' % (i))(input_data)
				x = nn.functional.relu(x)
				docking_output_list.append(x)
		
		if availabilities is None:
			availabilities = torch.ones(batch_size, len(input_list), dtype=torch.float, device=self.Device)
		else:
			availabilities = availabilities.float().to(self.Device)
		
		if selection_probabilities is None:
			selection_probabilities = torch.ones(batch_size, len(input_list), dtype=torch.float, device=self.Device)
		else:
			selection_probabilities = selection_probabilities.float().to(self.Device)
		
		selection_probabilities = torch.mul(selection_probabilities, availabilities)
		probability_sum = torch.sum(selection_probabilities, dim=-1, keepdim=True)
		selection_probabilities = torch.div(selection_probabilities, probability_sum)

		docking_output_stack = torch.stack(docking_output_list, dim=-1)

		modality_indices = torch.multinomial(selection_probabilities, num_samples=self.embracement_size, replacement=True)  # [batch_size, embracement_size]
		modality_toggles = nn.functional.one_hot(modality_indices, num_classes=num_modalities).float() # [batch_size, embracement_size, num_modalities]

		embracement_output_stack = torch.mul(docking_output_stack, modality_toggles)
		embracement_output = torch.sum(embracement_output_stack, dim=-1)

		return embracement_output
