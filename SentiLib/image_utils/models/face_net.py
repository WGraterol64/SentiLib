"""
	Implementation of https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch was taken as the basis for this implementation.
	The model implemented here is a VGG that due to the size of the input images (48x48) an fc of 512 is used.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torchvision.models.utils import load_state_dict_from_url
from torchvision.models import vgg19

cfg = {
	'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

model_urls = {
	'VGG11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
	'VGG13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
	'VGG16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
	'VGG19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

class ShortVGG(nn.Module):
	def __init__(self, vgg_name='VGG19', numclasses=8, pretrain=False):
		super(ShortVGG, self).__init__()
		self.Name = vgg_name
		self.features = self._make_layers(cfg[vgg_name])
		self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)
		self.classifier = nn.Linear(512, numclasses)
		if pretrain:
			self.load_pretrain()

	def forward(self, x):
		out = self.features(x)
		out = self.avgpool(out)
		out = out.view(out.size(0), -1)
		out = F.dropout(out, p=0.5, training=self.training)
		out_cl = self.classifier(out)
		return out_cl, out

	def _make_layers(self, cfg):
		layers = []
		in_channels = 3
		for x in cfg:
			if x == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			else:
				layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
									nn.BatchNorm2d(x),
									nn.ReLU(inplace=True)]
				in_channels = x
		
		return nn.Sequential(*layers)

	def load_pretrain(self):
		state_dict = load_state_dict_from_url(model_urls[self.Name])
		currstate = self.state_dict()
		ml, mm = 0, 0
		for name, param in state_dict.items():
			if name not in currstate:
				continue
			if isinstance(param, torch.nn.parameter.Parameter):
				param = param.data
			try:
				currstate[name].copy_(param)
				ml += 1
			except:
				print('missing', name)
				mm += 1
				pass
		self.load_state_dict(currstate)
		print('{} modules loaded and {} modules missing'.format(ml,mm))