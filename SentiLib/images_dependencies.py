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

from .image_utils.utils.dataset import Emotic_MultiDB, Rescale, RandomCrop, ToTensor, my_collate
from .image_utils.utils.traineval import train_step, eval
from .image_utils.utils.mat2py import get_skeleton_data
from .image_utils.models.fusion_model import MergeClass
from .image_utils.models.face_net import ShortVGG as VGG
from .image_utils.models.context_net import resnet18 as ABN
from .image_utils.models.skeleton_net import Model as DGCNN
from .image_utils.models.YOLOv3 import YOLOv3
from .image_utils.models.basic_HRnet import SimpleHRNet as HRnet

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
    'facial': 'PP_facevgg_best.pth',
    'bodily': 'PP_bodyabn_last.pth',
    'contextual': 'PP_contextabn_last.pth',
    'postural': 'PP_posedgcnn_ws_last.pth'}

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

    if len(faces) > 0:
        face, freg = faces[0][0], faces[0][1] # taking just the top(0) detected face
        fbbox = [freg[0],freg[0]+freg[2],freg[1],freg[1]+freg[3]]
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
        
        self.mode = None
        self.dataset_root = None
        self.multimodal, self.modality = None, None
        self.Model = None
        self.criterion, self.optimiser = None, None
        self.load_model()
    
    def load_model(self):
        if self.Arguments.unimodal:
            self.multimodal = False
            self.modality = self.Arguments.modality
        else:
            self.multimodal = True
            self.modality = 'all'

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
                f = open(self.Arguments.configuration,)
                model_configuration = json.load(f)
                f.close()
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
        curr_directory = os.path.dirname(os.path.realpath(__file__))
        yolo_class_path= curr_directory + "/image_utils/checkpoints/YOLO/coco.names"
        if use_tiny_yolo:
            yolo_model_def = curr_directory + "/image_utils/checkpoints/YOLO/yolov3-tiny.cfg"
            yolo_weights_path= curr_directory + "/image_utils/checkpoints/YOLO/YOLO-weights/yolov3-tiny.weights"
        else:
            yolo_model_def= curr_directory + "/image_utils/checkpoints/YOLO/yolov3.cfg"
            yolo_weights_path= curr_directory + "/image_utils/checkpoints/YOLO/YOLO-weights/yolov3.weights"
        detector = YOLOv3(model_def=yolo_model_def, class_path=yolo_class_path,
                    weights_path=yolo_weights_path, classes=('person',),
                    max_batch_size=16, device=self.device)
        detections = detector.predict_single(image)
        if detections is None:
            return []
        fa_model = build_model()
        cpw48_dir =  curr_directory + '/image_utils/checkpoints/hrnet_w48_384x288.pth'
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
        if image_data == []:
            print('Empty modalities')
            return [[False]*8]
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
            prediction, _ = self.Model.forward(tdata)
            print(prediction)
            prediction = np.greater(prediction.detach().cpu().numpy(), thresholds)
            write_line = ""
            write_line += sample['name'] + ': '
            for emotion, pred in zip(emotions,prediction):
                write_line += emotion + ':' + str(pred) +', '
            with open('results', 'a') as f:
                f.writelines(write_line)
                f.writelines('\n')
        # print('Inference concluded ...')
        # print(prediction)
        return prediction
    
    def start(self):
        input = cv2.imread(self.Arguments.inputfile)
        inference = self.final_inference(input)
        return inference
            
class Arguments:
    def __init__(self, 
                 inputfile,
                 cuda = 0, 
                 savename = 'default_savename', 
                 mode= 'inference', 
                 unimodal = False, 
                 modality= 'all', 
                 configuration= None,
                 unimodels= None,
                 pretrained= True,
                 multimodel= None,
                 threshold= None):
        curr_directory = os.path.dirname(os.path.realpath(__file__))

        if configuration is None:
            configuration = curr_directory + '/image_utils/configs/embracenet_plus.json'
        if unimodels is None:
            unimodels = curr_directory + '/image_utils/checkpoints/unimodals/'
        if multimodel is None:
            multimodel = curr_directory + '/image_utils/checkpoints/multimodal/PP_all_mergenew_last.pth'
        if threshold is None:
            threshold = curr_directory + '/image_utils/checkpoints/thresholds_validation.npy'
            
        self.cuda = cuda
        self.inputfile = inputfile
        self.unimodal = unimodal
        self.threshold = threshold
        self.unimodels = unimodels
        self.configuration = configuration
        self.pretrained = pretrained
        self.modality = modality
        self.multimodel = multimodel
        


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise Exception('Boolean value expected.')