import pickle
import json
import os
from openai import OpenAI
import time
from pprint import pprint
import random

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
    elif name == "CoT_few_shot_1":
        i = 0
        system_prompt = "You are a student asked to compare the naturalness of two paths and output which one is more natural, for the naturalness of ConceptNet research. You must eventually wrap your answer in $$ (e.g. $A$ or $B$)."
        for pairs in rows:
            user_prompt = f"""
<A> Avenue --[IsA]--> Street <--[RelatedTo]--> Car --[UsedFor]--> Fun
<B> Town --[AtLocation]--> Country <--[AtLocation]-- State <--[IsA]-- Office
(Break Down) 
<A>
"Avenue --[IsA]--> Street": Avenue is roughly a synonym of Street. We have the Seventh Avenue in Manhattan. It is a busy street.
"Street <--[RelatedTo]--> Car": Yeah indeed. Busy streets in city got so many cars.
"Car --[UsedFor]--> Fun": Emmmm...right. Cars can be used for travel and travel is fun. But cars is not always used for travel. Hmmmmm, a bit weird.
<B>
"Town --[AtLocation]--> Country": A town is in a country...? Nation? Oh, you mean the countryside area. Country road take me home...Ok, a bit weird but fair enough.
"Country <--[AtLocation]-- State": State at country...? Still the countryside? Oh, country here means nation now. And states are not those physics states but the states of the US, for example. Fair enough.
"State <--[IsA]-- Office": How is office a state? Office, state...is there an office mode and non-office mode? Super weird.
(Synthesis) Ok, seems I made more stops when analysing naturalness of B, I will choose A to be the more natural one indeed.
(Answer) $A$


<A> Office <--RelatedTo--> Day <--RelatedTo--> Sun <--RelatedTo--> Molecule
<B> Step <--RelatedTo--> Surface <--RelatedTo--> Level <--RelatedTo--> Hill
(Break Down)
<A>
"Office <--RelatedTo--> Day": Emmmm...Oh, office is usually open at day time. And probably we can say a work day is an office day. But the "related to" one is a bit weird.
"Day <--RelatedTo--> Sun": Yes! Sun appears at Day time! No issue.
"Sun <--RelatedTo--> Molecule": Hmmmm...Sun is not a molecule...How are they related? Those cosmic fantasy are in my head...Maybe our universe is a giant molecule? Oh, I think about a realistic connection——sun creates energy for molecules to undergo reactions. Hmmm but this relation is still a bit weird to me.
<B>
"Step <--RelatedTo--> Surface": Emmm...Oh, step can be a verb. One can step onto a flat surface, like climbing stairs. Fair enough.
"Surface <--RelatedTo--> Level": Yeah, I got several interpretations in my head. At surface level, blah blah. Also, the surface is level means the surface is flat. "Related to" is a bit ambiguous though.
"Level <--RelatedTo--> Hill": Emmm ok! Hills got levels (aka their heights). And some top of the hill is level (flat). Fair enough. 
(Synthesis) Ok, these two both got me stop for a short while. But the sun related to molecule one is really hard to draw realistic connection. I will choose B to be the more natural one indeed.
(Answer) $B$


<A> Paper <--[RelatedTo]-- Card <--[RelatedTo]-- Unit <--[IsA]-- Molecule
<B> Instrument --[Causes]--> Job <--[RelatedTo]-- Office <--[RelatedTo]--> Type
(Break Down) 
<A>
"Instrument --[Causes]--> Job": Emmmm...what? What does instruments mean? What type of gudgets? Music instrument? Musicians today are usually jobless though...And it is not causing a job to happen. Getting a job causes the buying of instrument.
"Job <--[RelatedTo]-- Office": Yeah very closely related! The future of all coders...
"Office <--[RelatedTo]--> Type": Emmmm....huh? What office? What type? Good office, bad office...But everything can have a type, which makes this relationship lame and strange.
<B>
"Paper <--[RelatedTo]-- Card": Yes. Paper can be used to make card. So they are somewhat related! Pocke cards are flying around my head now.
"Card <--[RelatedTo]-- Unit": Emmmm...what? Oh, maybe a card is a unit in a card-based game. Can a card be a measurable unit though...Emmm no. Yeah this relationship can work, but it is a bit strange. 
"Unit <--[IsA]-- Molecule": Yes! Molecule is indeed a unit of structures and life! Hopefully I have not forget about my secondary school biology knowledge.
(Synthesis) Ok, seems I made more stops when analysing naturalness of A, I will choose B to be the more natural one indeed.
(Answer) $B$


<A> Source <--RelatedTo--> Sun --UsedFor--> Life <--RelatedTo--> People
<B> Nation <--RelatedTo--> Land <--Desires-- Person <--IsA-- Job
(Break Down) 
<A>
"Source <--RelatedTo--> Sun": Emm yeah! Sun is the source of energy! They are related.
"Sun --UsedFor--> Life": Indeed, solar power is fantastic! Solar panels are flying in my head...Ohh yeah, and photosynthesis! 
"Life <--RelatedTo--> People": Well, "related to" is a bit ambiguous but...yeah, people care about their life.
<B>
"Nation <--RelatedTo--> Land": Yes! Land is crucial to the survival of a nation! Trump is such a crazy person that he even wants to buy Greenland. Crazy.
"Land <--Desires-- Person": Yes. Modern people do desire a land to live. And what the ancient governments want most is the land. Land is indeed something humans want a lot.
"Person <--IsA-- Job": Emmmm...what? Job is...a person? Is that a cartoon or something...Maybe the direction is wrong. But wait, person is also not a job what. Rideculous relation.
(Synthesis) Ok, the "Person <--IsA-- Job" one really itches me a lot. I will choose A to be the more natural one!
(Answer) $A$


<A> {pairs[0]}
<B> {pairs[1]}
(Break Down)
<A>
<B>
(Synthesis)
(Answer)
"""   
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
    elif name == "original_o3mini":
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
                    "model": "o3-mini",
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
            if len(content) < 4:
                if "<A>" in content:
                    label = "$A$"
                elif "<B>" in content:
                    label = "$B$"
                else:
                    label = random.choice(["$A$", "$B$"])
                    count += 1
            else:
                if "$A$" in content or "$a$" in content or "A$" in content or "A\n$" in content or "A\n\n$" in content or "$\text{A}$" in content or "\( A \)" in content:
                    label = "$A$"
                elif "$B$" in content or "$b$" in content or "B$" in content or "B\n$" in content or "B\n\n$" in content or "$\text{B}$" in content or "\( B \)" in content:
                    label = "$B$"
                else:
                    #print(content)
                    label = random.choice(["$A$", "$B$"])
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
    # prepare_data("CoT_few_shot_1", rows)
    # prepare_data("original_o3mini", rows)
# 
    # client = OpenAI()
    # with open("../data/finegrained/original_o3mini/request.jsonl", 'rb') as f:
    #     batch_input_file = client.files.create(file=f, purpose="batch")
    #     batch_input_file_id = batch_input_file.id
    #     print(f"Batch input file created: {batch_input_file}")
# 
    # job = client.batches.create(
    #     input_file_id=batch_input_file_id,
    #     endpoint="/v1/chat/completions",
    #     completion_window="24h",
    #     metadata={
    #         "description": "original_o3mini"
    #     }
    # )
    # print("Batch job:")
# # 
    # pprint(job)
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

    # training_data = list(filter(lambda x: x[2] <= 0.2 or x[2] >= 0.8, training_data))
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

