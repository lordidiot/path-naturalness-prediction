
from __future__ import division
import numpy as np
import torch, sys, os
from torch import nn, optim
from torch.autograd import Variable
from model import ChainEncoder, Predictor, PredictorSoftLabel
from dataset import Dataset
from multiprocessing import Pool

def train(features, fea_len, split_frac, out_file,
		  dataset_name='science', path_filename='paths.pkl', data_filename='answers.txt',
		  soft_label=False, new_path_format=False, random_seed=42):
	# Fix random seed for reproducibility
	np.random.seed(random_seed)	
	torch.manual_seed(random_seed)
	if isinstance(out_file, str):
		out_file = open(out_file, 'w')
	d = Dataset(dataset_name, features, split_frac, gpu, 
		path_filename=path_filename, data_filename=data_filename,
		soft_label=soft_label, new_path_format=new_path_format, random_seed=random_seed)
	enc = ChainEncoder(d.get_v_fea_len(), d.get_e_fea_len(), fea_len, 'last')
	# New training pipeline for experiments that use win-rate soft labels: set soft_label=True
	predictor = Predictor(fea_len) if not soft_label else PredictorSoftLabel(fea_len)
	loss = nn.NLLLoss() if not soft_label else nn.BCELoss()
	if gpu:
		enc.cuda()
		predictor.cuda()
		loss.cuda()

	optimizer = optim.Adam(list(enc.parameters())+list(predictor.parameters()))

	print('training')
	enc.train()
	test_chain_A, test_chain_B, test_y = d.get_test_pairs()
	test_y = test_y.data.cpu().numpy()
	for train_iter in range(4000):
		chains_A, chains_B, y = d.get_train_pairs(1000)
		enc.zero_grad()
		predictor.zero_grad()
		output_A = enc(chains_A)
		output_B = enc(chains_B)
		softmax_output = predictor(output_A, output_B)
		if soft_label:
			# only use first path's softmax value
			softmax_output = softmax_output[:, 0]
		loss_val = loss(softmax_output, y)
		loss_val.backward()
		optimizer.step()

		enc.zero_grad()
		predictor.zero_grad()
		with torch.no_grad():
			enc.eval()
			output_test_A = enc(test_chain_A)
			output_test_B = enc(test_chain_B)
			softmax_output = predictor(output_test_A, output_test_B).data.cpu().numpy()
			test_y_pred = softmax_output.argmax(axis=1)
			cur_acc = (test_y_pred==test_y).sum() / len(test_y)
			print('iter: ', train_iter, ' test acc:', cur_acc)
			out_file.write('%f\n'%cur_acc)
			if train_iter%50==0:
				torch.save(enc.state_dict(), 
					'ckpt/%i_encoder.model'%train_iter)
				torch.save(predictor.state_dict(), 
					'ckpt/%i_predictor.model'%train_iter)
		enc.train()
	torch.save(enc.state_dict(), 'ckpt/final_encoder.model')
	torch.save(predictor.state_dict(), 'ckpt/final_predictor.model')
	out_file.close()

##################################
# [!] TRAINING CONFIGURATION [!] #
##################################
RANDOM_SEED = 42
DATASET_NAME = "science"
DATA_FILENAME = "human_train_answers.txt"
PATH_FILENAME = "paths.pkl"
SOFT_LABEL = False
NEW_PATH_FORMAT = False
FEATURE_LEN = 20
SPLIT_FRAC = 0.8 / (0.8 + 0.1) # train / (train + test), eval on 0.1
FEATURES = ['v_enc_dim300', 'v_freq_freq', 'v_deg', 'v_sense', 'e_vertexsim', 'e_dir', 'e_rel', 'e_weightsource', 'e_sense']

gpu = False
train(FEATURES, FEATURE_LEN, SPLIT_FRAC, 'train.log',
	  dataset_name=DATASET_NAME, path_filename=PATH_FILENAME, data_filename=DATA_FILENAME,
	  soft_label=SOFT_LABEL, new_path_format=NEW_PATH_FORMAT, random_seed=RANDOM_SEED)
