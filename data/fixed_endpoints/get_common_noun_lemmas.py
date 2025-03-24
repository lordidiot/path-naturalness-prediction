# original wordlist: https://www.wordfrequency.info/samples/lemmas_60k_words.txt

words = []

with open('data/fixed-endpoints/words.txt', 'r') as f:
    lines = f.readlines()
    prev = ''
    for line in lines[9:]:
        line_list = line.split('\t')
        lemma = line_list[1]
        pos = line_list[2]
        if pos != 'n':
            continue
        if lemma == prev:
            continue
        words.append(lemma)
        prev = lemma
    f.close()

print(len(words))
with open('data/fixed-endpoints/common_noun_lemmas.txt', 'w') as f:
    for word in words:
        f.write(word.replace('-', '_') + '\n')

