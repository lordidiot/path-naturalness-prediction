input_file = "../../data/finegrained/softlabel.txt"
output_file = "../../data/finegrained/softlabel2.txt"

with open(input_file, "r") as fin, open(output_file, "w") as fout:
    for line in fin:
        clean_line = line.replace("cs4248/", "")
        fout.write(clean_line)
        print("!")