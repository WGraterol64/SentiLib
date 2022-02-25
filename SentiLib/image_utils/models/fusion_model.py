"""
	The models implemented here are the architectures to use the original EmbraceNets and the proposed EmbraceNet +.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.componets import WeightedSum, EmbraceNet

import numpy as np

class ModelOne(nn.Module):
	'''
		EembraceNet
	'''
	def __init__(self, num_classes, input_sizes, embrace_size, docker_architecture, finalouts, 
								device, use_ll, ll_config, trainable_probs):
		super(ModelOne, self).__init__()
		self.NClasses =  num_classes
		self.InputSize = len(input_sizes)
		self.Device = device
		self.EmbNet = EmbraceNet(input_sizes, embrace_size, docker_architecture, self.Device)
		self.FinalOut = finalouts
		self.UseLL = use_ll
		self.TrainableProbs = trainable_probs
		self.initProbabilities()
		if use_ll or num_classes != embrace_size:
			self.UseLL = True
			self.LL = self.gen_ll(ll_config, embrace_size)

	def gen_ll(self, config ,embrace_size):
		layers = []
		inC = embrace_size
		for x in config:
			if x == 'D':
				layers += [nn.Dropout()]
			elif x == 'R':
				layers += [nn.ReLU()]
			else:
				layers += [nn.Linear(inC, x)]
				inC = x

		return nn.Sequential(*layers)
	
	def initProbabilities(self):
		p = torch.ones(1, self.InputSize, dtype=torch.float)
		self.p = torch.div(p, torch.sum(p, dim=-1, keepdim=True)).to(self.Device)

		self.P = nn.Parameter(self.p, requires_grad=self.TrainableProbs)

	def forward(self, outputs1, outputs2, available):
		batch_size = outputs1[0].shape[0]
		availabilities = torch.ones(batch_size , self.InputSize, dtype=torch.float, device=self.Device)
		for i, av in enumerate(available):
			if av == 0.0:
				availabilities[:,i] = 0.0

		probabilities = torch.stack([self.p]*batch_size, dim=0).view(batch_size, self.InputSize)
		if self.FinalOut:
			out = self.EmbNet.forward(outputs2, availabilities, probabilities)
		else:
			out = self.EmbNet.forward(outputs1, availabilities, probabilities)
		if self.UseLL:
			outl = self.LL(out)
			return outl, out
		return out, None

class ModelNewFour(nn.Module):
	'''
		EmbraceNet +, which integrate three EmbraceNets and add a naive concatenation and a weighted sum
	'''
	def __init__(self, num_classes, input_sizes, final_input_sizes,
								embrace1_param, embrace2_param, embrace3_param, wsum_confg,
								device, trainable_probs, useffinal, use_ws, use_ll, ll_configs):
		super(ModelNewFour, self).__init__()
		self.NClasses =  num_classes
		self.InputSize = input_sizes
		self.FinalInputSize = final_input_sizes
		self.Device = device
		self.EmbNet1 = EmbraceNet(**embrace1_param)
		self.EmbNet2 = EmbraceNet(**embrace2_param)
		self.EmbNet3 = EmbraceNet(**embrace3_param)
		self.WeightedSum = WeightedSum(**wsum_confg)
		self.UseLL1 = use_ll[0]
		self.UseLL2 = use_ll[1]
		self.UseLL3 = use_ll[2]
		self.UseFinalsInFinal = useffinal
		self.UseWSum = use_ws
		self.TrainableProbs = trainable_probs
		self.initProbabilities()
		if self.UseLL1:
			self.LL1 = self.gen_ll(**ll_configs[0])
		if self.UseLL2:
			self.LL2 = self.gen_ll(**ll_configs[1])
		if self.UseLL3:
			self.LL3 = self.gen_ll(**ll_configs[2])

	def gen_ll(self, config ,embrace_size):
		layers = []
		inC = embrace_size
		for x in config:
			if x == 'D':
				layers += [nn.Dropout()]
			elif x == 'R':
				layers += [nn.ReLU()]
			else:
				layers += [nn.Linear(inC, x)]
				inC = x

		return nn.Sequential(*layers)
	
	def initProbabilities(self):
		p1 = torch.ones(1, self.InputSize, dtype=torch.float)
		p2 = torch.ones(1, self.InputSize, dtype=torch.float)
		p3 = torch.ones(1, self.FinalInputSize, dtype=torch.float)
		self.p1 = torch.div(p1, torch.sum(p1, dim=-1, keepdim=True)).to(self.Device)
		self.p2 = torch.div(p2, torch.sum(p2, dim=-1, keepdim=True)).to(self.Device)
		self.p3 = torch.div(p3, torch.sum(p3, dim=-1, keepdim=True)).to(self.Device)

		self.P1 = nn.Parameter(p1,requires_grad=self.TrainableProbs)
		self.P2 = nn.Parameter(p2,requires_grad=self.TrainableProbs)
		self.P3 = nn.Parameter(p3, requires_grad=self.TrainableProbs)

	def forward(self, outputs1, outputs2, available):
		batch_size = outputs1[0].shape[0]
		availabilities = torch.ones(batch_size , self.InputSize+4, dtype=torch.float, device=self.Device)
		for i, av in enumerate(available):
			if av == 0.0:
				availabilities[:,i] = 0.0
		
		probabilities1 = torch.stack([self.p1]*batch_size,dim=0).view(batch_size, self.InputSize)
		out1 = self.EmbNet1.forward(outputs1, availabilities[:,:-4], probabilities1)
		if self.UseLL1:
			out1 = self.LL1(out1)
		
		probabilities2 = torch.stack([self.p2]*batch_size,dim=0).view(batch_size, self.InputSize)
		out2 = self.EmbNet2.forward(outputs2, availabilities[:,:-4], probabilities2)
		if self.UseLL2:
			out2 = self.LL2(out2)

		wsout = self.WeightedSum.forward(torch.stack(outputs2, dim=1), availabilities[:,:-4])
		concat = torch.cat(outputs2, dim=-1)

		probabilities3 = torch.stack([self.p3]*batch_size, dim=0).view(batch_size, self.FinalInputSize)
		
		if not self.UseFinalsInFinal:
			availabilities[:, -1] = 0.0
		if not self.UseWSum:
			availabilities[:, -2] = 0.0

		out = self.EmbNet3.forward([out1,out2,wsout,concat], availabilities[:, 4:], probabilities3)
		
		if self.UseLL3:
			out = self.LL3(out)
		
		return out, (out1, out2, wsout)

class MergeClass():
	'''
		This is a wrapper class for the true trainable fusion class
	'''
	def __init__(self, models={}, config={}, device=torch.device('cpu'),
								labels={}, self_embeding=False, debug_mode=False):
		'''
			models				: dictionary with the models already loaded and in eval mode
			config				: dictionary with the parameters to define the merge module
			device				: torch device
			dataset				: dataset already loaded
			tags					: dictionary with the tags used
			self_embeding : boolean, if true then embed heuristic is used
			debug_mode		: bolean, if true, show various final and average informations
		'''
		self.Modalities = models
		self.MergeConfig = config
		self.Device = device
		self.MergeModel = self.get_model(self.MergeConfig)
		self.Classes = labels
		self.SelfEmbeddin = self_embeding

	def get_model(self, config):
		type = config['type']
		if type == 1:
			return ModelOne(**config['parameters']).to(self.Device)
		elif type == 5:
			return ModelNewFour(**config['parameters']).to(self.Device)
		else:
			raise NameError('type {} is not supported yet'.format(type))
		# models 2, 3 and 4 was discarded
	
	def parameters(self):
		return self.MergeModel.parameters()
	
	def train(self):
		self.MergeModel.train()
	
	def eval(self):
		self.MergeModel.eval()
	
	def state_dict(self):
		return self.MergeModel.state_dict()
	def load_state_dict(self, dict):
		self.MergeModel.load_state_dict(dict)

	def forward(self, data):
		availables = [1.0] *4
		fb, _, mb = self.Modalities['body'].forward(data['body'])
		fc, _, mc = self.Modalities['context'].forward(data['context'])
		middle_out = [mb[3], mc[3]]
		final_out = [fb, fc]
		if data['face'].sum().item() != 0.0:
			ff, mf = self.Modalities['face'].forward(data['face'])
			middle_out += [mf]
			final_out += [ff]
		else:
			availables[2] = 0.0
			middle_out += [mc[3]]
			final_out += [fc]
		if data['joint'].sum().item() != 0.0:
			fs, ms = self.Modalities['pose'].forward((data['joint'],data['bone']),0)
			ms = torch.cat((ms[0], ms[1]), dim=-1)
			middle_out += [ms]
			final_out += [fs]
		else:
			availables[3] = 0.0
			middle_out += [mc[3]]
			final_out += [fc]

		out, middle = self.MergeModel.forward(middle_out, final_out, availables)

		return out, middle