import random
from tqdm import tqdm

def _sample_pairs_fairly_from_entries(entries: list, num_occurrences_per_entry: int):
	'''
	from {entries}, sample non-repeating pairs
	so that each entry appears the same number of {num_occurrences_per_entry} times
	'''
	random.shuffle(entries)
	num_pairs_to_sample = len(entries) * num_occurrences_per_entry // 2

	possible_positions = [[] for _ in range(num_pairs_to_sample)]
	samples = []

	for e in tqdm(entries):
		count = 0
		while count < num_occurrences_per_entry:
			if len(possible_positions) < 5:
				# handle edge case where last path must form pair with itself
				has_other_path_to_pair = False
				for remaining_pos in possible_positions:
					if len(remaining_pos) == 0 or remaining_pos[0] != e:
						has_other_path_to_pair = True
						break
				if not has_other_path_to_pair:
					break
			pos = random.randint(0, len(possible_positions) - 1)
			item = possible_positions[pos]
			if len(item) == 1 and item[0] == e:
				continue
			count += 1
			item.append(e)
			if len(item) == 2:
				samples.append(tuple(item))
				del possible_positions[pos]
	
	return samples


def generate_soft_label_pair_samples(dir_name: str, file_name: str, out: str, num_occurrences_per_entry: int, mode = 'rr'):
	'''
	from paths in file_name, generate pairs so that each path occurs {num_occurrences_per_entry} times
	in total generating {num_occurrences_per_entry * len(paths) // 2} pairs
	'''
	entries = []
	with open(f'{dir_name.strip("/")}/{file_name}') as f:
		for line in f:
			entries.append(line.strip())
	
	samples = _sample_pairs_fairly_from_entries(entries, num_occurrences_per_entry)

	def rr_tuple_to_soft_label(t):
		a = t[0].split('_')[0]
		score_a = float(t[0].split('_')[1])
		b = t[1].split('_')[0]
		score_b = float(t[1].split('_')[1])
		score_a_norm = score_a / (score_a + score_b) if not (score_a == 0 and score_b == 0) else 0.5
		return f'{a}_{b}_{score_a_norm}'

	if mode == 'rr':
		samples = list(map(rr_tuple_to_soft_label, samples))
		with open(f'{dir_name.strip("/")}/{out if out else file_name + "_softlabels"}', 'w') as f:
			for sample in samples:
				f.write(f'{sample}\n')
	elif mode == 'elo':
		return NotImplementedError
	else:
		return NotImplementedError

def rr_ans_to_soft_label(dir_name: str, file_name: str, out: str):
	'''
	from round robin comparisons in {file_name}, compile winning probability for each path and output as soft label in {out}
	'''
	match_record = {} # id -> [wins, total]
	with open(f'{dir_name.strip("/")}/{file_name}') as f:
		for line in f:
			a, b, winner = line.strip().split('_')
			match_record.setdefault(a, [0, 0])
			match_record.setdefault(b, [0, 0])
			match_record[a][1] += 1
			match_record[b][1] += 1
			if winner == a:
				match_record[a][0] += 1
			else:
				match_record[b][0] += 1
	with open(f'{dir_name.strip("/")}/{out if out else file_name + "_softlabels"}', 'w') as f:
		for id, (wins, total) in match_record.items():
			score = wins / total
			f.write(f'{id}_{score}\n')

if __name__ == '__main__':
	# rr_ans_to_soft_label('data/money', 'rr_answers.txt', 'rr_answers_softlabel.txt')
	generate_soft_label_pair_samples('data/money', 'rr_answers_softlabel.txt', 'rr_answers_pairwise_softlabel.txt', 30, mode='rr')