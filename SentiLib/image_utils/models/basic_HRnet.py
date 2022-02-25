"""
	The original implementation is in https://github.com/stefanopini/simple-HRNet
	In our implementation, the use for several estimating people at the same time was eliminated
"""

import cv2
import numpy as np
import torch
from torchvision.transforms import transforms

from .utils.hrnet import HRNet
# from .utils.poseresnet import PoseResNet

class SimpleHRNet:
	"""
	SimpleHRNet class.
	The class provides a simple and customizable method to load the HRNet network, load the official pre-trained
	weights, and predict the human pose on single images.
	Multi-person support with the YOLOv3 detector is also included (and enabled by default).
	"""
	def __init__(self,
							 c,
							 nof_joints,
							 checkpoint_path,
							 model_name='HRNet',
							 resolution=(384, 288),
							 interpolation=cv2.INTER_CUBIC,
							 multiperson=True,
							 return_bounding_boxes=False,
							 max_batch_size=32,
							 device=torch.device("cuda")):
		"""
		Initializes a new SimpleHRNet object.
		HRNet (and YOLOv3) are initialized on the torch.device("device") and
		its (their) pre-trained weights will be loaded from disk.
		Args:
				c (int): number of channels (when using HRNet model) or resnet size (when using PoseResNet model).
				nof_joints (int): number of joints.
				checkpoint_path (str): path to an official hrnet checkpoint or a checkpoint obtained with `train_coco.py`.
				model_name (str): model name (HRNet or PoseResNet).
						Valid names for HRNet are: `HRNet`, `hrnet`
						Valid names for PoseResNet are: `PoseResNet`, `poseresnet`, `ResNet`, `resnet`
						Default: "HRNet"
				resolution (tuple): hrnet input resolution - format: (height, width).
						Default: (384, 288)
				interpolation (int): opencv interpolation algorithm.
						Default: cv2.INTER_CUBIC
				multiperson (bool): if True, multiperson detection will be enabled.
						This requires the use of a people detector (like YOLOv3).
						Default: True
				return_bounding_boxes (bool): if True, bounding boxes will be returned along with poses by self.predict.
						Default: False
				max_batch_size (int): maximum batch size used in hrnet inference.
						Useless without multiperson=True.
						Default: 16
				yolo_model_def (str): path to yolo model definition file.
						Default: "./models/detectors/yolo/config/yolov3.cfg"
				yolo_class_path (str): path to yolo class definition file.
						Default: "./models/detectors/yolo/data/coco.names"
				yolo_weights_path (str): path to yolo pretrained weights file.
						Default: "./models/detectors/yolo/weights/yolov3.weights.cfg"
				device (:class:`torch.device`): the hrnet (and yolo) inference will be run on this device.
						Default: torch.device("cpu")
		"""

		self.c = c
		self.nof_joints = nof_joints
		self.checkpoint_path = checkpoint_path
		self.model_name = model_name
		self.resolution = resolution  # in the form (height, width) as in the original implementation
		self.interpolation = interpolation
		self.multiperson = multiperson
		self.return_bounding_boxes = return_bounding_boxes
		self.max_batch_size = max_batch_size
		self.device = device

		if model_name in ('HRNet', 'hrnet'):
			self.model = HRNet(c=c, nof_joints=nof_joints)
		# elif model_name in ('PoseResNet', 'poseresnet', 'ResNet', 'resnet'):
		# 	self.model = PoseResNet(resnet_size=c, nof_joints=nof_joints)
		else:
			raise ValueError('Wrong model name.')

		checkpoint = torch.load(checkpoint_path, map_location=self.device)
		if 'model' in checkpoint:
			self.model.load_state_dict(checkpoint['model'])
		else:
			self.model.load_state_dict(checkpoint)

		if 'cuda' in str(self.device):
			print("device: 'cuda' - ", end="")

			if 'cuda' == str(self.device):
				# if device is set to 'cuda', all available GPUs will be used
				print("%d GPU(s) will be used" % torch.cuda.device_count())
				device_ids = None
			else:
				# if device is set to 'cuda:IDS', only that/those device(s) will be used
				print("GPU(s) '%s' will be used" % str(self.device))
				device_ids = [int(x) for x in str(self.device)[5:].split(',')]

			self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
		elif 'cpu' == str(self.device):
			print("device: 'cpu'")
		else:
			raise ValueError('Wrong device name.')

		self.model = self.model.to(device)
		self.model.eval()

		
		self.transform = transforms.Compose([transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

	def predict(self, image):
		"""
		Predicts the human pose on a single image or a stack of n images.
		Args:
				image (:class:`np.ndarray`):
						the image(s) on which the human pose will be estimated.
						image is expected to be in the opencv format.
						image can be:
								- a single image with shape=(height, width, BGR color channel)
								- a stack of n images with shape=(n, height, width, BGR color channel)
		Returns:
				:class:`np.ndarray`:
						a numpy array containing human joints for each (detected) person.
						Format:
								if image is a single image:
										shape=(# of people, # of joints (nof_joints), 3);  dtype=(np.float32).
								if image is a stack of n images:
										list of n np.ndarrays with
										shape=(# of people, # of joints (nof_joints), 3);  dtype=(np.float32).
						Each joint has 3 values: (y position, x position, joint confidence).
						If self.return_bounding_boxes, the class returns a list with (bounding boxes, human joints)
		"""
		if len(image.shape) == 3:
			return self._predict_single(image)
		else:
			raise ValueError('Wrong image format.')

	def _predict_single(self, image):
		# if not self.multiperson:
		old_res = image.shape
		if self.resolution is not None:
			image = cv2.resize(image,
												 (self.resolution[1], self.resolution[0]),  # (width, height)
												 interpolation=self.interpolation)

		images = self.transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).unsqueeze(dim=0)
		boxes = np.asarray([[0, 0, old_res[1], old_res[0]]], dtype=np.float32)  # [x1, y1, x2, y2]

		if images.shape[0] > 0:
			images = images.to(self.device)

			with torch.no_grad():
				if len(images) <= self.max_batch_size:
					out = self.model(images)
				else:
					out = torch.empty((images.shape[0], self.nof_joints, self.resolution[0] // 4, self.resolution[1] // 4),
														device=self.device)
					for i in range(0, len(images), self.max_batch_size):
						out[i:i + self.max_batch_size] = self.model(images[i:i + self.max_batch_size])

			out = out.detach().cpu().numpy()
			pts = np.empty((out.shape[0], out.shape[1], 3), dtype=np.float32)
			# For each human, for each joint: y, x, confidence
			for i, human in enumerate(out):
				for j, joint in enumerate(human):
					pt = np.unravel_index(np.argmax(joint), (self.resolution[0] // 4, self.resolution[1] // 4))
					# 0: pt_y / (height // 4) * (bb_y2 - bb_y1) + bb_y1
					# 1: pt_x / (width // 4) * (bb_x2 - bb_x1) + bb_x1
					# 2: confidences
					pts[i, j, 0] = pt[0] * 1. / (self.resolution[0] // 4) * (boxes[i][3] - boxes[i][1]) + boxes[i][1]
					pts[i, j, 1] = pt[1] * 1. / (self.resolution[1] // 4) * (boxes[i][2] - boxes[i][0]) + boxes[i][0]
					pts[i, j, 2] = joint[pt]

		else:
			pts = np.empty((0, 0, 3), dtype=np.float32)

		if self.return_bounding_boxes:
			return boxes, pts
		else:
			return pts