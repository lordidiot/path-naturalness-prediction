import pickle
import json
import os
from openai import OpenAI
import time
from pprint import pprint

def to_path(code, data):
    id = code[:-1]
    dir = code[-1]
    dir_map = {'f': 'forward', 'r': 'reverse'}
    dir_key = dir_map[dir]
    short = data[id][dir_key]['short']
    return short

def prepare_data(name, rows):
    os.makedirs(f"../data/finegrained/{name}", exist_ok=True)
    if name == "zero_shot":
        system_prompt = (
            "You are an expert in natural language understanding. Given two conceptual paths that connect "
            "words or ideas, your task is to evaluate which path sounds more natural, intuitive, or human-like in reasoning.\n\n"
            "A more natural path flows logically and smoothly in meaning, like how people would typically associate ideas.\n\n"
            "Only respond with one of:\n"
            '"A"\n'
            '"B"\n'
            "No need for any explanation. Just output A or B."
        )
        i = 0
        for pairs in rows:
            user_prompt = f"Question <A> {pairs[0]}\n<B> {pairs[1]}\nAnswer:"
            data = {
                "custom_id": f"R{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                }
            }
            with open(f"../data/finegrained/{name}/request.jsonl", "a") as f:
                json.dump(data, f)
                f.write("\n")
            i += 1

    elif name == "one_shot":
        system_prompt = (
            "You are an expert in natural language understanding. Given two conceptual paths that connect "
            "words or ideas, your task is to evaluate which path sounds more natural, intuitive, or human-like in reasoning.\n\n"
            "A more natural path flows logically and smoothly in meaning, like how people would typically associate ideas.\n\n"
            "Only respond with one of:\n"
            '"A"\n'
            '"B"\n'
            "No need for any explanation. Just output A or B.\n\n"
            "Question: <A> 'Lead <--Synonym--> Take Distinct <--From−−> Give <--RelatedTo--> Poison'\n"
            "<B> 'Lead --HasProperty--> Toxic <--Related To--> Lethal <--RelatedTo--> Poison'”\n"
            "Answer:B"
        )
        i = 0
        for pairs in rows:
            user_prompt = f"Question <A> {pairs[0]}\n<B> {pairs[1]}\nAnswer:"
            data = {
                "custom_id": f"R{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                }
            }
            with open(f"../data/finegrained/{name}/request.jsonl", "a") as f:
                json.dump(data, f)
                f.write("\n")
            i += 1

    elif name == "CoT_zero_shot":
        system_prompt = (
            "You are an expert in natural language understanding. Given two conceptual paths that connect "
            "words or ideas, your task is to evaluate which path sounds more natural, intuitive, or human-like in reasoning.\n\n"
            "Think of the question in the following criteria: Is it too long? (-1 if yes) Is there unneeded shifts in word class? (-1 if yes) Is there illogical word-word transition? (-1 if yes). A path is better if deduction of score is lower than the other one.  \n\n"
            "Only respond with one of:\n"
            '"A"\n'
            '"B"\n'
            "No need for any explanation. Just output A or B."
        )
        i = 0
        for pairs in rows:
            user_prompt = f"Question <A> {pairs[0]}\n<B> {pairs[1]}\nAnswer:"
            data = {
                "custom_id": f"R{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                }
            }
            with open(f"../data/finegrained/{name}/request.jsonl", "a") as f:
                json.dump(data, f)
                f.write("\n")
            i += 1

    elif name == "CoT_one_shot":
        system_prompt = (
            "You are an expert in natural language understanding. Given two conceptual paths that connect "
            "words or ideas, your task is to evaluate which path sounds more natural, intuitive, or human-like in reasoning.\n\n"
            "Think of the question in the following criteria: Is it too long? (-1 if yes) Is there unneeded shifts in word class? (-1 if yes) Is there illogical word-word transition? (-1 if yes). A path is better if deduction of score is lower than the other one.  \n\n"
            "Only respond with one of:\n"
            '"A"\n'
            '"B"\n'
            "No need for any explanation. Just output A or B.\n\n"
            "Question: <A> 'Lead <--Synonym--> Take Distinct <--From−−> Give <--RelatedTo--> Poison'\n"
            "<B> 'Lead --HasProperty--> Toxic <--Related To--> Lethal <--RelatedTo--> Poison'”\n"
            "(Implicit thinking) A isn't very long (no deduction), has unnecessary change of word class (deduction -1), and has illogical transition (deduction -1). So the score is -2. B isn't very long (no deduction), doesn't have unnecessary change of word class (no deduction), and has no illogical transition (no deduction). So the score is 0. Thus, B is better. I need to output B. \n"
            "Answer:B"
            ""
        )
        i = 0
        for pairs in rows:
            user_prompt = f"Question <A> {pairs[0]}\n<B> {pairs[1]}\nAnswer:"
            data = {
                "custom_id": f"R{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                }
            }
            with open(f"../data/finegrained/{name}/request.jsonl", "a") as f:
                json.dump(data, f)
                f.write("\n")
            i += 1
    elif name == "few-shot":
        system_prompt = (
            "You are an expert in natural language understanding. Given two conceptual paths that connect "
            "words or ideas, your task is to evaluate which path sounds more natural, intuitive, or human-like in reasoning.\n\n"
            "A more natural path flows logically and smoothly in meaning, like how people would typically associate ideas.\n\n"
            "Only respond with one of:\n"
            '"A"\n'
            '"B"\n'
            "No need for any explanation. Just output A or B.\n\n"
            "Question: <A> 'Lead <--Synonym--> Take Distinct <--From−−> Give <--RelatedTo--> Poison'\n"
            "<B> 'Lead --HasProperty--> Toxic <--Related To--> Lethal <--RelatedTo--> Poison'”\n"
            "Answer:B\n"
            "Question: <A> 'Knowledge <--HasA-- Book <--RelatedTo--> Paper'\n"
            "<B> 'Knowledge <--HasA-- Book <--RelatedTo--> Restaurant'”\n"
            "Answer:A\n"
            "Question: <A> 'Wave --IsA--> Fluctuation <--RelatedTo--> Brainwave --DerivedFrom--> Brain'\n"
            "<B> 'Adult <--RelatedTo--> A <--RelatedTo--> The <--RelatedTo--> English'\n"
            "Answer:A\n"
        )
        i = 0
        for pairs in rows:
            user_prompt = f"Question <A> {pairs[0]}\n<B> {pairs[1]}\nAnswer:"
            data = {
                "custom_id": f"R{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                }
            }
            with open(f"../data/finegrained/{name}/request.jsonl", "a") as f:
                json.dump(data, f)
                f.write("\n")
            i += 1
    elif name == "original":
        i = 0
        for pairs in rows:
            user_prompt = f"""
Which of the following paths connecting two concepts is the most natural?

A) {pairs[0]}
B) {pairs[1]}

Explain first, then wrap your answer in $ (e.g. $A$ or $B$).
"""
            data = {
                "custom_id": f"R{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "user", "content": user_prompt}
                    ]
                }
            }
            with open(f"../data/finegrained/{name}/request.jsonl", "a") as f:
                json.dump(data, f)
                f.write("\n")
            i += 1

        

        
def extract_contents_from_jsonl(jsonl_file):
    contents = []
    with open(jsonl_file, "r") as f:
        count = 0
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip empty lines
            record = json.loads(line)
            content = record["response"]["body"]["choices"][0]["message"]["content"]
            if "$A$" in content:
                label = "$A$"
            elif "$B$" in content:
                label = "$B$"
            else:
                label = "$B$" # default to $B$ if neither is found, in 13800 instances, there are < 10 occurances
                count += 1
            contents.append(label)
        print(count)
    return contents

def main():
    rows = []
    with open("../data/science/llm_answers.txt", "r") as file:
        for line in file:
            parts = line.strip().split("_")
            pair = [parts[0], parts[1]]
            rows.append(pair)
    training_data = list(map(lambda x: [x[0], x[1], 0], rows))
    with open("../data/science/paths.pkl", "rb") as file:
        data = pickle.load(file)
    rows = list(map(lambda x: [to_path(x[0], data), to_path(x[1], data)], rows))
    
    # prepare_data("zero_shot", rows)
    # prepare_data("one_shot", rows)
    # prepare_data("CoT_zero_shot", rows)
    # prepare_data("CoT_one_shot", rows)
    # prepare_data("few-shot", rows)
    # prepare_data("original", rows)

    #client = OpenAI()
    #with open("../data/finegrained/original/request.jsonl", 'rb') as f:
    #    batch_input_file = client.files.create(file=f, purpose="batch")
    #    batch_input_file_id = batch_input_file.id
    #    print(f"Batch input file created: {batch_input_file}")
#
    #job = client.batches.create(
    #    input_file_id=batch_input_file_id,
    #    endpoint="/v1/chat/completions",
    #    completion_window="24h",
    #    metadata={
    #        "description": "original"
    #    }
    #)
    #print("Batch job:")
# #
    #pprint(job)

    # contents_zero_shot_1 = extract_contents_from_jsonl("../data/finegrained/zero_shot/zero_shot_output_1.jsonl")
    # contents_zero_shot_2 = extract_contents_from_jsonl("../data/finegrained/zero_shot/zero_shot_output_2.jsonl")
    # contents_zero_shot_3 = extract_contents_from_jsonl("../data/finegrained/zero_shot/zero_shot_output_3.jsonl")
    # contents_one_shot_1 = extract_contents_from_jsonl("../data/finegrained/one_shot/one_shot_output_1.jsonl")
    # contents_one_shot_2 = extract_contents_from_jsonl("../data/finegrained/one_shot/one_shot_output_2.jsonl")
    # contents_one_shot_3 = extract_contents_from_jsonl("../data/finegrained/one_shot/one_shot_output_3.jsonl")
    # contents_few_shot_1 = extract_contents_from_jsonl("../data/finegrained/few-shot/few-shot_output_1.jsonl")
    # contents_few_shot_2 = extract_contents_from_jsonl("../data/finegrained/few-shot/few-shot_output_2.jsonl")
    # contents_few_shot_3 = extract_contents_from_jsonl("../data/finegrained/few-shot/few-shot_output_3.jsonl")
    # contents_CoT_zero_shot_1 = extract_contents_from_jsonl("../data/finegrained/CoT_zero_shot/CoT_zero_shot_output_1.jsonl")
    # contents_CoT_zero_shot_2 = extract_contents_from_jsonl("../data/finegrained/CoT_zero_shot/CoT_zero_shot_output_2.jsonl")
    # contents_CoT_zero_shot_3 = extract_contents_from_jsonl("../data/finegrained/CoT_zero_shot/CoT_zero_shot_output_3.jsonl")
    # contents_CoT_one_shot_1 = extract_contents_from_jsonl("../data/finegrained/CoT_one_shot/CoT_one_shot_output_1.jsonl")
    # contents_CoT_one_shot_2 = extract_contents_from_jsonl("../data/finegrained/CoT_one_shot/CoT_one_shot_output_2.jsonl")
    # contents_CoT_one_shot_3 = extract_contents_from_jsonl("../data/finegrained/CoT_one_shot/CoT_one_shot_output_3.jsonl")

    contents = []
    for i in range(1,31):
        contents.append(extract_contents_from_jsonl(f"../data/finegrained/original/{i}_output.jsonl"))

    for j in range(len(training_data)):
        for i in range(30):
            if contents[i][j] == "$A$":
                training_data[j][2] += 1

    for i in range(len(training_data)):
        training_data[i][2] = training_data[i][2] / 30

    training_data = list(map(lambda x: x[0]+"_"+x[1]+"_"+f"{x[2]}", training_data))
    output_file = "../data/finegrained/softlabel.txt"
    
    with open(output_file, "w") as f:
        for path in training_data:
            f.write(path + "\n")

    #contents_1 = extract_contents_from_jsonl("../data/finegrained/original/output_1.jsonl")
    #for i in range(len(training_data)):
    #    if contents_1[i] == "$A$":
    #        training_data[i][2] += 1
    #for i in range(len(training_data)):
        #if contents_zero_shot_1[i] == "A":
        #    training_data[i][2] += 1
        #if contents_zero_shot_2[i] == "A":
        #    training_data[i][2] += 1
        #if contents_zero_shot_3[i] == "A":
        #    training_data[i][2] += 1
        #if contents_one_shot_1[i] == "A":
        #    training_data[i][2] += 1
        #if contents_one_shot_2[i] == "A":
        #    training_data[i][2] += 1
        #if contents_one_shot_3[i] == "A":
        #    training_data[i][2] += 1
        #if contents_few_shot_1[i] == "A":
        #    training_data[i][2] += 1
        #if contents_few_shot_2[i] == "A":
        #    training_data[i][2] += 1
        #if contents_few_shot_3[i] == "A":
        #    training_data[i][2] += 1
        #if contents_CoT_zero_shot_1[i] == "A":
        #    training_data[i][2] += 1
        #if contents_CoT_zero_shot_2[i] == "A":
        #    training_data[i][2] += 1
        #if contents_CoT_zero_shot_3[i] == "A":
        #    training_data[i][2] += 1
        #if contents_CoT_one_shot_1[i] == "A":
        #    training_data[i][2] += 1
        #if contents_CoT_one_shot_2[i] == "A":
        #    training_data[i][2] += 1
        #if contents_CoT_one_shot_3[i] == "A":
        #    training_data[i][2] += 1
    #for i in range(len(training_data)):
    #    training_data[i][2] = training_data[i][2] / 1
    #training_data = list(map(lambda x: x[0]+"_"+x[1]+"_"+f"{x[2]}", training_data))
    #output_file = "../data/finegrained/softlabel.txt"
    #
    #with open(output_file, "w") as f:
    #    for path in training_data:
    #        f.write(path + "\n")
    

if __name__ == "__main__":
    main()

