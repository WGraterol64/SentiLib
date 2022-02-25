import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import torch.nn as nn

class DatasetIEMOCAP(Dataset):
    def __init__(self, classes, FaceR, AudioR, TextR, method='avg', mode='train', transform=None):
        super(DatasetIEMOCAP, self).__init__()
        self.Data = {}
        self.DataKeys = []
        # self.Face = True
        # self.Audio = True
        # self.Text = True
        self.Transform = transform
        self.Classes = classes
        self.Mode = mode
        self.Method = method
        self.loadData(FaceR, AudioR, TextR)

    def loadData(self, face_results, audio_results, text_results):

        iterable_keys = []
        if face_results is not None:
            LFks = list(face_results.keys())
            iterable_keys = LFks
        else:
            LFks = []

        if audio_results is not None:
            LAks = list(audio_results.keys())
            iterable_keys = LAks if len(LFks) < len(LAks) else LFks
        else:
            LAks = []

        if text_results is not None:
            LTks = list(text_results.keys())
            iterable_keys = LTks if len(LAks) < len(LTks) else LAks
        else:
            LTks = []


        for k in iterable_keys:
            
            if k in LFks:
                FD = face_results[k][0]
            else:
                FD = None
            if k in LAks:
                AD = audio_results[k][0]
            else:
                AD = None
            if k in LTks:
                TD = text_results[k][0]
            else:
                TD = None

            self.Data[k] = (text_results[k][1], FD, AD, TD)
        self.DataKeys = list(self.Data.keys())
  
    def __len__(self):
        return len(self.DataKeys)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        avs = np.ones(3)
        label = self.Data[self.DataKeys[idx]][0]
        face = self.Data[self.DataKeys[idx]][1]
        if face is None:
            avs[0] = 0.
            face = np.zeros(self.Classes['number'])
        
        audio = self.Data[self.DataKeys[idx]][2]
        if audio is None:
            avs[1] = 0.
            audio = np.zeros(self.Classes['number'])
        
        text = self.Data[self.DataKeys[idx]][3]
        if text is None:
            avs[2] = 0.
            text = np.zeros(self.Classes['number'])

        sample = {'face': face,
                  'audio': audio,
                  'text': text,
                  'label': label, 
                  'availabilities':avs,
                  'name': self.DataKeys[idx]}
        if self.Transform:
            sample = self.Transform(sample)
        return sample
    
    def make_shuffle(self):
        random.shuffle(self.DataKeys)

class EmbraceNet(nn.Module):
    def __init__(self, device, input_size_list, embracement_size=256, bypass_docking=False):
        super(EmbraceNet, self).__init__()

        self.device = device
        self.input_size_list = input_size_list
        self.embracement_size = embracement_size
        self.bypass_docking = bypass_docking
        if (not bypass_docking):
            for i, input_size in enumerate(input_size_list):
                setattr(self, 'docking_%d' % (i), nn.Linear(input_size, embracement_size))

    def forward(self, input_list, availabilities=None, selection_probabilities=None):
        # check input data
        assert len(input_list) == len(self.input_size_list)
        num_modalities = len(input_list)
        batch_size = input_list[0].shape[0]
        # docking layer
        docking_output_list = []
        if (self.bypass_docking):
            docking_output_list = input_list
        else:
            for i, input_data in enumerate(input_list):
                # print(i)
                # print(input_data)
                x = getattr(self, 'docking_%d' % (i))(input_data)
                # print('pass')
                x = nn.functional.relu(x)
                docking_output_list.append(x)
        # check availabilities
        if (availabilities is None):
            availabilities = torch.ones(batch_size, len(input_list), dtype=torch.float, device=self.device)
        else:
            availabilities = availabilities.float()
        # adjust selection probabilities
        if (selection_probabilities is None):
            selection_probabilities = torch.ones(batch_size, len(input_list), dtype=torch.float, device=self.device)
        selection_probabilities = torch.mul(selection_probabilities, availabilities)

        probability_sum = torch.sum(selection_probabilities, dim=-1, keepdim=True)
        selection_probabilities = torch.div(selection_probabilities, probability_sum)
        # stack docking outputs
        docking_output_stack = torch.stack(docking_output_list, dim=-1)  # [batch_size, embracement_size, num_modalities]
        # embrace
        modality_indices = torch.multinomial(selection_probabilities, num_samples=self.embracement_size, replacement=True)  # [batch_size, embracement_size]
        modality_toggles = nn.functional.one_hot(modality_indices, num_classes=num_modalities).float()  # [batch_size, embracement_size, num_modalities]

        embracement_output_stack = torch.mul(docking_output_stack, modality_toggles)
        embracement_output = torch.sum(embracement_output_stack, dim=-1)  # [batch_size, embracement_size]

        return embracement_output

class Wrapper(nn.Module):
    def __init__(self, device, n_classes=6, size_list=[6,6,6],
                embracesize=100, bypass_docking=False):
        super(Wrapper, self).__init__()
        self.NClasses = n_classes
        self.Embrace = EmbraceNet(device=device,
                                input_size_list=size_list,
                                embracement_size=embracesize,
                                bypass_docking=bypass_docking)
        self.classifier = False
        if embracesize != n_classes:
            self.classifier = True
            # setattr(self, 'docking_%d' % (i), nn.Linear(input_size, embracement_size))
            self.clf = nn.Sequential(nn.Linear(embracesize, n_classes),
                                    nn.Softmax(dim=-1))

    def forward(self, face, audio, text, avs):
        out = self.Embrace([face, audio, text], availabilities=avs)
        if self.classifier:
            out = self.clf(out)
        return out

class FusionTransformer(object):
  def __init__(self, modename):
    self.mode = modename

  def __call__(self, sample):
    facedata, audiodata, textdata = sample['face'], sample['audio'], sample['text']
    label, avs, name = sample['label'], sample['availabilities'], sample['name']

    # facedata = torch.flatten(torch.from_numpy(facedata))
    # audiodata = F.softmax(torch.from_numpy(audiodata),dim=-1)
    facedata = torch.from_numpy(facedata)
    audiodata = torch.from_numpy(audiodata)
    textdata = torch.from_numpy(textdata)
    avs = torch.from_numpy(avs)
    label = np.asarray(label)

    return {'face': facedata.float(),
            'audio': audiodata.float(),
            'text': textdata.float(),
            'label': torch.from_numpy(label).long(),
            'availabilities': avs.float(),
            'name': name}
    
def my_collate(batch):
    batch = filter(lambda img: img is not None, batch)
    return default_collate(list(batch))