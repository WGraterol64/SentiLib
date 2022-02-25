import os
import argparse
import numpy as np
import json
import cv2

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms

from tqdm import tqdm
from deepface import DeepFace
from deepface.detectors.RetinaFaceWrapper import build_model, detect_face

from utils.dataset import Emotic_MultiDB, Rescale, RandomCrop, ToTensor, my_collate
from utils.traineval import train_step, eval
from utils.mat2py import get_skeleton_data
from models.fusion_model import MergeClass
from models.face_net import ShortVGG as VGG
from models.context_net import resnet18 as ABN
from models.skeleton_net import Model as DGCNN
from models.YOLOv3 import YOLOv3
from models.basic_HRnet import SimpleHRNet as HRnet

original_cats = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval',
				'Disconnection', 'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem',
				'Excitement', 'Fatigue', 'Fear', 'Happiness', 'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity',
				'Suffering', 'Surprise', 'Sympathy', 'Yearning']

modal_dirs = ['context', 'face', 'person', 'posture-bones', 'posture-joints']

new_labels = {'len': 8, 'cat': {
	'joy': ['Excitement', 'Happiness', 'Peace', 'Affection', 'Pleasure'],
	'trust': ['Confidence', 'Esteem', 'Sympathy'],
	'fear': ['Disquietment', 'Embarrassment', 'Fear'],
	'surprice': ['Doubt/Confusion', 'Surprise'],
	'sadness': ['Pain', 'Sadness', 'Sensitivity', 'Suffering'],
	'disgust': ['Aversion', 'Disconnection', 'Fatigue', 'Yearning'],
	'anger': ['Anger', 'Annoyance', 'Disapproval'],
	'anticipation': ['Anticipation', 'Engagement']}
			}

unimodels_default = {
	'facial': '',
	'bodily': '',
	'contextual': '',
	'postural': ''}

def get_weighted_random_sampler(root_dir='Emotic_MultiDB', annotation_dir='Annotations', mode='train',
								modality='label', takeone=True, modals_dirs=modal_dirs, categories=original_cats,
								relabel=True, new_labeles=new_labels):
	dataset = Emotic_MultiDB(root_dir, annotation_dir, mode, modality, takeone, modals_dirs, categories)
	if relabel:
		dataset.relabeled(new_labels)
	
	target = []
	for i in range(len(dataset)):
		try:
			target += [np.argmax(dataset[i]['label'])]
		except:
			pass
	target = np.asarray(target)
	class_sample_count = np.unique(target, return_counts=True)[1]
	weight = 1. / class_sample_count
	samples_weight = weight[target]
	
	samples_weight = torch.from_numpy(samples_weight).double()
	sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
	return sampler

def extract_data(ori_img, bbox, pose_model, fa_model):
	if len(bbox) == 0:
		return None
	context = ori_img.copy()
	body = context[bbox[1]:bbox[3],bbox[0]:bbox[2]].copy()
	cn = body.shape

	pose = pose_model.predict(body)

	body = cv2.resize(body,(256,256))
	faces = detect_face(fa_model, body, align=False)

	# fbbox, _ = fa_model.detect(body,threshold=0.5, scale=1.0) # v0
	# face, freg = DeepFace.functions.detect_face(body, detector_backend='retinaface', align=False) v2
	
	# if len(fbbox) != 0: # v0
	# 	fbbox = np.round(fbbox[0]).astype('int32')
	# 	face = body[fbbox[1]:fbbox[3],fbbox[0]:fbbox[2]].copy()
	# 	body[fbbox[1]:fbbox[3],fbbox[0]:fbbox[2]] = np.zeros(face.shape)
	# else:
	# 	face = None
	# if face not is None: #v2
	# 	fbbox = [rfeg[0],rfeg[0]+rfeg[2],rfeg[1],rfeg[1]+rfeg[3]]
	# 	body[fbbox[2]:fbbox[3],fbbox[0]:fbbox[1]] = np.zeros(face.shape)

	if len(faces) > 0:
		face, freg = faces[0][0], faces[0][1] # taking just the top(0) detected face
		fbbox = [rfeg[0],rfeg[0]+rfeg[2],rfeg[1],rfeg[1]+rfeg[3]]
		body[fbbox[2]:fbbox[3],fbbox[0]:fbbox[1]] = np.zeros(face.shape)
	else:
		face = None

	context[bbox[1]:bbox[3],bbox[0]:bbox[2]] = np.zeros(cn)
	joints, bones = get_skeleton_data(pose[0])
	
	return (context, body, face, joints, bones)

class Processor:
	def __init__(self, args):
		self.Arguments = args
		if self.Arguments.cuda < 0:
			self.device = torch.device('cpu')
		elif self.Arguments.cuda == 0:
			self.device = torch.device('cuda')
		else:
			self.device = torch.device('cuda:' + str(self.Arguments.cuda))  # chek it works
		
		if not os.path.exists('./checkpoints'):
			os.mkdir('checkpoints')
		
		self.mode = None
		self.dataset_root = None
		self.multimodal, self.modality = None, None
		self.test_db, self.train_db, self.val_db = None, None, None
		self.Model = None
		self.criterion, self.optimiser = None, None
		self.epoch, self.last_epoch, self.batch_size = 0, 36, 16
		self.train_acc, self.val_acc, self.train_loss, self.val_loss = None, None, None, None
		self.train_sampler, self.val_sampler = None, None
		self.model_saved_name, self.saving_step = self.Arguments.savename, 4
		
		self.my_collate = my_collate
		self.load_data()
		self.load_model()
	
	def load_data(self):
		self.mode = self.Arguments.mode
		self.dataset_root = self.Arguments.dataset
		if self.Arguments.unimodal:
			self.multimodal = False
			self.modality = self.Arguments.modality
		else:
			self.multimodal = True
			self.modality = 'all'
		
		if self.mode == "test":
			self.test_db = Emotic_MultiDB(root_dir=self.dataset_root,
										annotation_dir='Annotations',
										mode='test',
										modality=self.modality,
										modals_dirs=modal_dirs,
										categories=original_cats,
										transform=transforms.Compose([Rescale(224, 224, 48),
																		ToTensor()])
										)
			self.test_db.relabeled(new_labels)
		else:
			self.train_db = Emotic_MultiDB(root_dir=self.dataset_root,
										annotation_dir='Annotations',
										mode='train',
										modality=self.modality,
										modals_dirs=modal_dirs,
										categories=original_cats,
										transform=transforms.Compose([Rescale(256, 256, 56),
																		RandomCrop(224, 224, 48),
																		ToTensor()])
										)
			self.train_db.relabeled(new_labels)
			self.val_db = Emotic_MultiDB(root_dir=self.dataset_root,
										annotation_dir='Annotations',
										mode='val',
										modality=self.modality,
										modals_dirs=modal_dirs,
										categories=original_cats,
										transform=transforms.Compose([Rescale(256, 256, 56),
																	RandomCrop(224, 224, 48),
																	ToTensor()])
										)
			self.val_db.relabeled(new_labels)
		
		if self.Arguments.oversample:
			self.train_sampler = get_weighted_random_sampler(root_dir=self.dataset_root, annotation_dir='Annotations')
			self.val_sampler = get_weighted_random_sampler(root_dir=self.dataset_root, annotation_dir='Annotations',
														mode='val')
	
	def load_model(self):
		if self.multimodal:
			with open(self.Arguments.configuration) as jf:
				model_configuration = json.load(jf)
			model_configuration = self.change_device(model_configuration)
			
			loaded = torch.load(self.Arguments.unimodels + unimodels_default['facial'])
			face_model = VGG('VGG19', 8).to(self.device)
			face_model.load_state_dict(loaded['model_state_dict'])
			
			loaded = torch.load(self.Arguments.unimodels + unimodels_default['bodily'])
			body_model = ABN(num_classes=8)
			body_model = body_model.to(self.device)
			body_model.load_state_dict(loaded['model_state_dict'])
			
			loaded = torch.load(self.Arguments.unimodels + unimodels_default['contextual'])
			context_model = ABN(num_classes=8)
			context_model = context_model.to(self.device)
			context_model.load_state_dict(loaded['model_state_dict'])
			
			loaded = torch.load(self.Arguments.unimodels + unimodels_default['postural'])
			pose_model = DGCNN().to(self.device)
			pose_model.load_state_dict(loaded['model_state_dict'])
			del loaded
			
			uni_models = {
				'body': body_model.eval(),
				'context': context_model.eval(),
				'face': face_model.eval(),
				'pose': pose_model.eval()
			}
			self.Model = MergeClass(uni_models, model_configuration, self.device)
		else:
			if self.Arguments.configuration:
				model_configuration = json.load(self.Arguments.configuration)
				model_configuration = self.change_device(model_configuration)
				
				if self.modality == 'face':
					try:
						self.Model = VGG(**model_configuration)
					except:
						print("Error to instantiate model configuration, default configuration is used instead.")
						self.Model = VGG('VGG19', 8)
				elif self.modality == 'body' or self.modality == 'context':
					try:
						self.Model = ABN(**model_configuration)
					except:
						print("Error to instantiate model configuration, default configuration is used instead.")
						self.Model = ABN(num_classes=8)
				elif self.modality == 'pose':
					try:
						self.Model = DGCNN(**model_configuration)
					except:
						print("Error to instantiate model configuration, default configuration is used instead.")
						self.Model = DGCNN()
			else:
				raise Exception("A model configuration isn't defined")
		
		self.criterion = nn.BCEWithLogitsLoss()
		self.optimiser = Adam(self.Model.parameters(), lr=0.001, weight_decay=5e-4)
		
		if self.Arguments.pretrained:
			loaded = torch.load(self.Arguments.multimodel)
			self.Model.load_state_dict(loaded['model_state_dict'])
			self.optimiser.load_state_dict(loaded['optimizer_state_dict'])
			self.epoch = loaded['epoch']
			self.train_acc = loaded['train_acc']
			self.val_acc = loaded['val_acc']
			self.train_loss = loaded['train_loss']
			self.val_loss = loaded['val_loss']
			del loaded
	
	def change_device(self, config):
		for k in config:
			if isinstance(config[k], dict):
				config[k] = self.change_device(config[k])
			if k == "device":
				config[k] = self.device
		return config
	
	def get_data_modalities(self, image, use_tiny_yolo=False):
		yolo_class_path="checkpoints/YOLO/coco.names"
		if use_tiny_yolo:
			yolo_model_def ="checkpoints/YOLO/yolov3-tiny.cfg"
			yolo_weights_path="checkpoints/YOLO/YOLO-weights/yolov3-tiny.weights"
		else:
			yolo_model_def="checkpoints/YOLO/yolov3.cfg"
			yolo_weights_path="checkpoints/YOLO/YOLO-weights/yolov3.weights"
		detector = YOLOv3(model_def=yolo_model_def, class_path=yolo_class_path,
					weights_path=yolo_weights_path, classes=('person',),
					max_batch_size=16, device=self.device)
		detections = detector.predict_single(image)

		# fa_model = insightface.model_zoo.get_model('retinaface_r50_v1')
		# fa_model.prepare(ctx_id = 0, nms=0.4)
		fa_model = build_model()
		cpw48_dir = 'checkpoints/hrnet_w48_384x288.pth'
		pose_model = HRnet(48, 17, cpw48_dir, multiperson=False, max_batch_size=2)

		all_data = []
		for i,(x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(detections):
			x1, x2 = int(round(x1.item())), int(round(x2.item()))
			y1, y2 = int(round(y1.item())), int(round(y2.item()))
			bbox = [x1, y1, x2, y2]
			data = extract_data(image, bbox, pose_model, fa_model)
			name = self.Arguments.inputfile[:-4] +'_'+ str(i).zfill(3)
			ctxnp = cv2.resize(data[0], (224,224))
			bodnp = cv2.resize(data[1], (224,224))
			if data[2] is None or len(data[2])==0:
				facnp = np.zeros(16)
			else:
				facnp = cv2.resize(data[2], (48,48))
			if data[3] is None or len(data[3])==0:
				joinp = np.zeros(16)
				bonnp = np.zeros(16)
			else:
				joinp = data[3]
				bonnp = data[4]
			all_data.append({'name': name,
					'context': ctxnp,
					'body': bodnp,
					'face': facnp,
					'joint':joinp,
					'bone':bonnp})
		return all_data
	
	def final_inference(self, image):
		thresholds = np.load(self.Arguments.threshold)
		emotions = new_labels['cat'].keys()
		image_data = self.get_data_modalities(image)
		for idx, sample in enumerate(image_data):
			tdata = dict()
			tdata['context'] = torch.from_numpy(sample['context'].transpose((2, 0, 1))).unsqueeze_(0).float().to(self.device)
			tdata['body'] = torch.from_numpy(sample['body'].transpose((2, 0, 1))).unsqueeze_(0).float().to(self.device)
			try:
				tdata['face'] = torch.from_numpy(sample['face'].transpose((2, 0, 1))).unsqueeze_(0).float().to(self.device)
				tdata['joint'] = torch.from_numpy(sample['joint']).unsqueeze_(0).float().to(self.device)
				tdata['bone'] = torch.from_numpy(sample['bone']).unsqueeze_(0).float().to(self.device)
			except:
				tdata['face'] = torch.from_numpy(sample['face']).unsqueeze_(0).float().to(self.device)
				tdata['joint'] = torch.from_numpy(sample['joint']).unsqueeze_(0).float().to(self.device)
				tdata['bone'] = torch.from_numpy(sample['bone']).unsqueeze_(0).float().to(self.device)
			prediction = self.Model.forward(tdata)
			prediction = np.greater(prediction.detach().numpy(), thresholds)
			write_line = ""
			write_line += sample['name'] + ': '
			for emotion, pred in zip(emotions,prediction):
				write_line += emotion + ':' + str(pred) +', '
			with open('results', 'a') as f:
				f.writelines(write_line)
				f.writelines('\n')
		print('Inference concluded ...')
	
	def start(self):
		if self.mode == 'train':
			c_maxacc = 0
			for epoch in range(self.epoch, self.epoch):
				c_maxacc = train_step(Model=self.Model, dataset_t=self.train_db, dataset_v=self.val_db,
									bsz=self.batch_size, Loss=self.criterion, optimizer=self.optimiser,
									collate=self.my_collate, epoch=epoch, tsampler=self.train_sampler,
									vsampler=self.val_sampler, last_epoch=self.last_epoch, modal=self.modality,
									device=self.device, debug_mode=False, tqdm=tqdm, train_loss=list,
									train_map=list, val_loss=list, val_map=list, maxacc=c_maxacc,
									step2save=self.saving_step, checkpointdir='', model_name=self.model_saved_name)
		
		elif self.mode == 'test':
			mAP = eval(Model=self.Model, dataset=self.test_db, bsz=self.batch_size, collate=self.my_collate,
					epoch=0, modal=self.modality, device=self.device, tqdm=tqdm)
			print('The mean AP is', mAP)

		elif self.mode == 'inference':
			input = cv2.imread(self.Arguments.inputfile)
			inference = self.final_inference(input)
			

def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise Exception('Boolean value expected.')


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-u", "--unimodal", action="store_true", help="specify the kind of running")
	parser.add_argument("-t", "--modality", type=str, help="if unimodal, specify the modality")
	parser.add_argument("-p", "--pretrained", action="store_true", help="pre trained models will be used")
	parser.add_argument("-n", "--unimodel", type=str, help="if unimodal and pretrain, give the model path")
	parser.add_argument("-u", "--unimodels", type=str, help="if multimodal and pretrain, give the models folder path")
	parser.add_argument("-m", "--multimodel", type=str, help="if multimodal and pretrain, give the multimodel path")
	parser.add_argument("-o", "--mode", type=str, help="specify if it is train of test")
	parser.add_argument("-d", "--dataset", type=str, help="folder of the dataset")
	parser.add_argument("-c", "--configuration", type=str, help="filename with the model config")
	parser.add_argument("-g", "--cuda", type=int, help="id of the cuda device, -1 if it no use", default=-1)
	parser.add_argument("-s", "--savename", type=str, help="name to save into checkpoints folder")
	parser.add_argument("-v", "--oversample", action="store_true", help="if oversample will be used")
	parser.add_argument("-i", "--inputfile", type=str, help="input image path for inference")
	parser.add_argument("-h", "--threshold", type=str, help="thresholds npy file path for inference") 
	
	arg = parser.parse_args()
	
	processor = Processor(arg)
	processor.start()
