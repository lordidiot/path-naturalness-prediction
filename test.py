file_1 = "data/finegrained/softlabel.txt"
file_2 = "data/science/human_eval_line_numbers.txt"

def remove_lines_from_file(file1_path, file2_path):
    # Read file1 (the main file)
    with open(file1_path, 'r', encoding='utf-8') as f1:
        file1_lines = f1.readlines()

    # Read file2 (the line numbers to remove)
    with open(file2_path, 'r', encoding='utf-8') as f2:
        line_numbers = [int(line.strip()) for line in f2 if line.strip().isdigit()]

    # Filter out any line numbers that are out of range
    max_lines = len(file1_lines)
    valid_line_numbers = [ln for ln in line_numbers if 1 <= ln <= max_lines]

    # Rebuild file1 content without the lines from file2
    updated_lines = [
        line for (idx, line) in enumerate(file1_lines, start=1)
        if idx not in valid_line_numbers
    ]

    return updated_lines

if __name__ == "__main__":
    # Perform the line removal
    updated_content = remove_lines_from_file(file_1, file_2)

    # Option 1: Print to stdout
    # print("".join(updated_content))

    # Option 2: Overwrite the original file1 or write to a new file:
    with open(file_1, 'w', encoding='utf-8') as out:
        out.writelines(updated_content)
    
    print(f"Removed lines specified in {file_2} from {file_1}.")