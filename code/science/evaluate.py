
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
             path_filename="paths.pkl", data_filename="answers.txt", soft_label=False):
	d = Dataset('money', features, 0, gpu, path_filename=path_filename, data_filename=data_filename, soft_label=soft_label)
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

	# When evaluating on hard label dataset, use this
    # print(evaluate(features, feature_len, encoder_path, predictor_path,
    #                path_filename="paths.pkl", data_filename="answers.txt", soft_label=False))
    
	# When evaluating on soft label dataset, use this
    print(evaluate(features, feature_len, encoder_path, predictor_path,
                   path_filename="paths.pkl", data_filename="answers.txt", soft_label=False))

if __name__ == '__main__':
    main()
