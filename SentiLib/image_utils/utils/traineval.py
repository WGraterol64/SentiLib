"""
	There are the functions for carry out the training, the evaluation, the model saving, and the AP calculator.
	The AP calculator is taken from https://github.com/Tandon-A/emotic/blob/master/Colab_train_emotic.ipynb.
"""

import os
import numpy as np
from time import sleep

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import average_precision_score, precision_recall_curve

def savemodel(epoch, model_dict, opt_dict, losstrain, acctrain, lossval, accval,
							save_dir, modelname, save_name):
	model_saved_name = os.path.join(save_dir,modelname + save_name +'.pth')
	torch.save({'epoch':epoch,
							'train_loss':losstrain,
							'train_acc':acctrain,
							'val_loss':lossval,
							'val_acc':accval,
							'model_state_dict':model_dict,
							'optimizer_state_dict':opt_dict},
						 model_saved_name)
	print('Model {} saved'.format(model_saved_name))

def test_AP(cat_preds, cat_labels, n_classes=8):
	n_classes=cat_labels.shape[0]
	ap = np.zeros(n_classes, dtype=np.float32)
	for i in range(n_classes):
		ap[i] = average_precision_score(cat_labels[i, :], cat_preds[i, :])
	ap[np.isnan(ap)] = 0.0
	print ('AveragePrecision: {} |{}| mAP: {}'.format(ap, ap.shape[0], ap.mean()))
	return ap.mean()

def train(Model, train_dataset, Loss, optimizer, val_dataset, bsz=32,
					collate=None, train_sampler=None, val_sampler=None, epoch=0,
					modal='all', device=torch.device('cpu'), debug_mode=False, tqdm=None):
	Model.train()
	if collate is not None:
		loader = tqdm(DataLoader(train_dataset, batch_size=bsz, num_workers=0, sampler=train_sampler, collate_fn=collate),
									unit='batch')
	else:
		loader = tqdm(DataLoader(train_dataset, batch_size=bsz, num_workers=0, sampler=train_sampler,),
									unit='batch')
	loader.set_description("{} Epoch {}".format(train_dataset.Mode, epoch + 1))
	loss_values = []
	predictions, labeles = [], []
	for batch_idx, batch_sample in enumerate(loader):
		with torch.no_grad():
			if modal == 'all':
				sample = dict()
				sample['context'] = batch_sample['context'].to(device)
				sample['body'] = batch_sample['body'].to(device)
				sample['face'] = batch_sample['face'].to(device)
				sample['joint'] = batch_sample['joint'].to(device)
				sample['bone'] = batch_sample['bone'].to(device)
			elif modal == 'pose':
				sample = (batch_sample['joint'].to(device),
									batch_sample['bone'].to(device))
			else:
				sample = batch_sample[modal].to(device)
			
			label = batch_sample['label'].to(device)

		optimizer.zero_grad()
		if modal =='pose':
			output, _ = Model.forward(sample, 0)
			predictions += [output[i].to('cpu').data.numpy() for i in range(output.shape[0])]
			loss = Loss(output, label)
		elif modal == 'face':
			output, _ = Model.forward(sample)
			predictions += [output[i].to('cpu').data.numpy() for i in range(output.shape[0])]
			loss = Loss(output, label)
		elif modal == 'body' or modal == 'context':
			per_outs, att_outs, _ = Model.forward(sample)
			predictions += [per_outs[i].to('cpu').data.numpy() for i in range(per_outs.shape[0])]
			loss = (Loss(att_outs, label)) + (Loss(per_outs, label))
		elif modal == 'all':
			output, _ = Model.forward(sample)
			predictions += [output[i].to('cpu').data.numpy() for i in range(output.shape[0])]
			loss = Loss(output, label)

		loss.backward()
		optimizer.step()

		labeles += [label[i].to('cpu').data.numpy() for i in range(label.shape[0])]
		
		loss_values.append(loss.item())
		loader.set_postfix(loss=loss.item())
		sleep(0.1)
	
	train_gloss = np.mean(loss_values)
	train_mAP = test_AP(np.asarray(predictions).T, np.asarray(labeles).T)

	if collate is not None:
		loader = tqdm(DataLoader(val_dataset, batch_size=bsz, num_workers=0, sampler=val_sampler, collate_fn=collate),
									unit='batch')
	else:
		loader = tqdm(DataLoader(val_dataset, batch_size=bsz, num_workers=0, sampler=val_sampler,),
									unit='batch')
	loader.set_description("{} Epoch {}".format(val_dataset.Mode, epoch + 1))
	loss_values = []
	predictions, labeles = [], []
	
	Model.eval()
	with torch.no_grad():
		for batch_idx, batch_sample in enumerate(loader):
			if modal == 'all':
				sample = dict()
				sample['context'] = batch_sample['context'].to(device)
				sample['body'] = batch_sample['body'].to(device)
				sample['face'] = batch_sample['face'].to(device)
				sample['joint'] = batch_sample['joint'].to(device)
				sample['bone'] = batch_sample['bone'].to(device)
			elif modal == 'pose':
				sample = (batch_sample['joint'].to(device),
									batch_sample['bone'].to(device))
			else:
				sample = batch_sample[modal].to(device)

			label = batch_sample['label'].to(device)

			if modal =='pose':
				output, _ = Model.forward(sample, 0)
				predictions += [output[i].to('cpu').data.numpy() for i in range(output.shape[0])]
				loss = Loss(output, label)
			elif modal == 'face':
				output, _ = Model.forward(sample)
				predictions += [output[i].to('cpu').data.numpy() for i in range(output.shape[0])]
				loss = Loss(output, label)
			elif modal == 'body' or modal == 'context':
				per_outs, att_outs, _ = Model.forward(sample)
				predictions += [per_outs[i].to('cpu').data.numpy() for i in range(per_outs.shape[0])]
				loss = (Loss(att_outs, label)) + (Loss(per_outs, label))
			elif modal == 'all':
				output, _ = Model.forward(sample)
				predictions += [output[i].to('cpu').data.numpy() for i in range(output.shape[0])]
				loss = Loss(output, label)
			labeles += [label[i].to('cpu').data.numpy() for i in range(label.shape[0])]
			
			loss_values.append(loss.item())
			loader.set_postfix(loss=loss.item())
			sleep(0.1)
	val_gloss = np.mean(loss_values)
	val_mAP = test_AP(np.asarray(predictions).T, np.asarray(labeles).T)

	if debug_mode:
		print ('- Mean training loss: {:.4f} ; epoch {}'.format(train_gloss, epoch+1))
		print ('- Mean validation loss: {:.4f} ; epoch {}'.format(val_gloss, epoch+1))
		print ('- Mean training mAP: {:.4f} ; epoch {}'.format(train_mAP, epoch+1))
		print ('- Mean validation mAP: {:.4f} ; epoch {}'.format(val_mAP, epoch+1))
	return train_gloss, train_mAP, val_gloss, val_mAP

def eval(Model, dataset, bsz=32, test_sampler=None, collate=None, epoch=0, modal='all',
				 device=torch.device('cpu'), debug_mode=False, tqdm=None):
	Model.eval()
	if collate is not None:
		loader = tqdm(DataLoader(dataset, batch_size=bsz, num_workers=0, sampler=test_sampler, collate_fn=collate),
									unit='batch')
	else:
		loader = tqdm(DataLoader(dataset, batch_size=bsz, num_workers=0, sampler=test_sampler),
									unit='batch')
	loader.set_description("{} Epoch {}".format(dataset.Mode, epoch + 1))
	predictions, labeles = [], []
	for batch_idx, batch_sample in enumerate(loader):
		sample = dict()
		with torch.no_grad():
			if modal == 'all':
				sample = dict()
				sample['context'] = batch_sample['context'].to(device)
				sample['body'] = batch_sample['body'].to(device)
				sample['face'] = batch_sample['face'].to(device)
				sample['joint'] = batch_sample['joint'].to(device)
				sample['bone'] = batch_sample['bone'].to(device)
				output, _ = Model.forward(sample)
				predictions += [output[i].to('cpu').data.numpy() for i in range(output.shape[0])]
			elif modal == 'pose':
				sample = (batch_sample['joint'].to(device),
									batch_sample['bone'].to(device))
				output, _ = Model.forward(sample,0)
				predictions += [output[i].to('cpu').data.numpy() for i in range(output.shape[0])]
			else:
				sample = batch_sample[modal].to(device)
				if modal == 'face':
					output, _ = Model.forward(sample)
				else:
					output, _, _ = Model.forward(sample)
				predictions += [output[i].to('cpu').data.numpy() for i in range(output.shape[0])]
			
			label = batch_sample['label']
		
		labeles += [label[i].data.numpy() for i in range(label.shape[0])]

	mAP = test_AP(np.asarray(predictions).T, np.asarray(labeles).T)
	return mAP, predictions, labeles

def train_step(Model, dataset_t, dataset_v, bsz, Loss, optimizer, collate, epoch,
			   tsampler, vsampler,
			   last_epoch, modal, device, debug_mode, tqdm, train_loss, train_map,
			   val_loss, val_map, maxacc, step2val, step2save, checkpointdir, model_name):
	
	tl, ta, vl, va = train(Model=Model, train_dataset=dataset_t, Loss=Loss, optimizer=optimizer,
													val_dataset=dataset_v, bsz=bsz, collate=collate, train_sampler=tsampler,
													val_sampler=vsampler, epoch=epoch, modal=modal,
													device=device, debug_mode=debug_mode, tqdm=tqdm)
	train_loss[epoch] = tl
	train_map[epoch] = ta
	val_loss[epoch] = vl
	val_map[epoch] = va
	
	if ta > maxacc:
		maxacc = ta
		savemodel(epoch=epoch,
							model_dict=Model.state_dict(),
							opt_dict=optimizer.state_dict(),
							losstrain=tl,	acctrain=ta,
							lossval=tl,	accval=ta,
							save_dir=checkpointdir, modelname=model_name, save_name='_best')

	if (epoch+1) % step2save == 0 or (epoch+1) == last_epoch:
		savemodel(epoch=epoch,
							model_dict=Model.state_dict(),
							opt_dict=optimizer.state_dict(),
							losstrain=tl,	acctrain=ta,
							lossval=tl,	accval=ta,
							save_dir=checkpointdir, modelname=model_name, save_name='_last')
	return maxacc


def get_thresholds(cat_preds, cat_labels, saving=False):
	n_cats = cat_labels.shape[0]
	thresholds = np.zeros(n_cats, dtype=np.float32)
	for i in range(n_cats):
		p, r, t = precision_recall_curve(cat_labels[i, :], cat_preds[i, :])
		# print(p,r,t)
		for k in range(len(p)):
			if p[k] == r[k]:
				thresholds[i] = t[k]
				break
	if saving:
		np.save('thresholds.npy', thresholds)
	return thresholds