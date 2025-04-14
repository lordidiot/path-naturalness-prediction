from __future__ import division

import numpy as np
import torch, sys, os
from torch import nn, optim
from torch.autograd import Variable
from model import ChainEncoder, Predictor, PredictorSoftLabel
from dataset import Dataset
from multiprocessing import Pool

gpu = False
def evaluate(d: Dataset, fea_len, encoder_path, predictor_path,
			 soft_label=False, random_seed=42):
	# Fix random seed for reproducibility
	np.random.seed(random_seed)	
	torch.manual_seed(random_seed)
	# d = Dataset(dataset_name, features, 0, gpu,
	# path_filename=path_filename, data_filename=data_filename, soft_label=soft_label, new_path_format=new_path_format)
	enc = ChainEncoder(d.get_v_fea_len(), d.get_e_fea_len(), fea_len, 'last')
	enc.load_state_dict(torch.load(encoder_path))
	predictor = Predictor(fea_len) if not soft_label else PredictorSoftLabel(fea_len)
	predictor.load_state_dict(torch.load(predictor_path))
	enc.eval()
	predictor.eval()
	if gpu:
		enc.cuda()
		predictor.cuda()
	test_chain_A, test_chain_B, test_y = d.get_test_pairs()
	test_y = test_y.data.cpu().numpy()
	with torch.no_grad():
		output_test_A = enc(test_chain_A)
		output_test_B = enc(test_chain_B)
		softmax_output = predictor(output_test_A, output_test_B).data.cpu().numpy()
		test_y_pred = softmax_output.argmax(axis=1)
		cur_acc = (test_y_pred==test_y).sum() / len(test_y)
		return cur_acc

###################################j
# [!] EVALUATION CONFIGURATION [!] #
####################################
DATASET_NAME = "money"
DATA_FILENAME = "answers.txt"
PATH_FILENAME = "paths.pkl"
SOFT_LABEL = False
NEW_PATH_FORMAT = False
FEATURE_LEN = 20
FEATURES = ['v_enc_dim300', 'v_freq_freq', 'v_deg', 'v_sense', 'e_vertexsim', 'e_dir', 'e_rel', 'e_weightsource', 'e_sense']

def main():
	if len(sys.argv) < 3:
		print(f"Usage: {sys.argv[0]} <encoder.model> <predictor.model>")
		return
	encoder_path, predictor_path = sys.argv[1:3]

	dataset = Dataset(DATASET_NAME, FEATURES, 0, gpu,
		path_filename=PATH_FILENAME, data_filename=DATA_FILENAME,
		soft_label=SOFT_LABEL, new_path_format=NEW_PATH_FORMAT)

	if encoder_path == "all" and predictor_path == "all":
		with open("./test.log", "w") as f:
			epoc = 0
			while epoc < 4000:
				encoder_path = f"ckpt/{epoc}_encoder.model"
				predictor_path = f"ckpt/{epoc}_predictor.model"
				print(f"Evaluating {encoder_path} and {predictor_path}")
				acc = evaluate(dataset, FEATURE_LEN, encoder_path, predictor_path,
							   soft_label=SOFT_LABEL)
				f.write(f'{acc}\n')
				epoc += 50
			encoder_path = "ckpt/final_encoder.model"
			predictor_path = "ckpt/final_predictor.model"
			acc = evaluate(dataset, FEATURE_LEN, encoder_path, predictor_path,
						   soft_label=SOFT_LABEL)
			f.write(f'{acc}\n')
			return

	# When evaluating on hard label dataset, use this
	print(evaluate(dataset, FEATURE_LEN, encoder_path, predictor_path,
				   soft_label=SOFT_LABEL))
	
	# When evaluating on soft label dataset, use this
	# print(evaluate(features, feature_len, encoder_path, predictor_path,
	#                path_filename="paths.pkl", data_filename="abc.txt", soft_label=True, new_path_format=False))

if __name__ == '__main__':
	main()
