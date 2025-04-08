from __future__ import division

import numpy as np
import torch, sys, os
from torch import nn, optim
from torch.autograd import Variable
from model import ChainEncoder, Predictor, PredictorSoftLabel
from dataset import Dataset
from multiprocessing import Pool

gpu = False
def evaluate(features, fea_len, encoder_path, predictor_path,
			path_filename="paths.pkl", data_filename="answers.txt", soft_label=False, new_path_format=False):
	d = Dataset('money', features, 0, gpu,
	path_filename=path_filename, data_filename=data_filename, soft_label=soft_label, new_path_format=new_path_format)
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

def main():
	# Removed 'e_srank_rel', 'e_trank_rel'
	features = ['v_enc_dim300', 'v_freq_freq', 'v_deg', 'v_sense', 'e_vertexsim',
		'e_dir', 'e_rel', 'e_weightsource', 'e_sense']
	feature_len = 20

	if len(sys.argv) < 3:
		print(f"Usage: {sys.argv[0]} <encoder.model> <predictor.model>")
		return
	encoder_path, predictor_path = sys.argv[1:3]

	if encoder_path == "all" and predictor_path == "all":
		with open("./test.log", "w") as f:
			epoc = 0
			while epoc < 4000:
				encoder_path = f"ckpt/{epoc}_encoder.model"
				predictor_path = f"ckpt/{epoc}_predictor.model"
				print(f"Evaluating {encoder_path} and {predictor_path}")
				acc = evaluate(features, feature_len, encoder_path, predictor_path,
						 path_filename="paths.pkl", data_filename="answers.txt", soft_label=False)
				f.write(f'{acc}\n')
				epoc += 50
			encoder_path = "ckpt/final_encoder.model"
			predictor_path = "ckpt/final_predictor.model"
			acc = evaluate(features, feature_len, encoder_path, predictor_path,
						 path_filename="paths.pkl", data_filename="answers.txt", soft_label=False)
			f.write(f'{acc}\n')
			return
				  
	print(evaluate(features, feature_len, encoder_path, predictor_path,
				   path_filename="paths.pkl", data_filename="answers.txt", soft_label=False))
	
	# evaluate model trained with soft label data
	# print(evaluate(features, feature_len, encoder_path, predictor_path,
	#                path_filename="paths.pkl", data_filename="rr_answers_pairwise_softlabel.txt", soft_label=True))

if __name__ == '__main__':
	main()
