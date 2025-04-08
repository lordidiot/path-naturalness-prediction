import numpy as np
import matplotlib.pyplot as plt

input_file = "../../data/finegrained/softlabel.txt"

with open(input_file, "r") as f:
    lines = f.readlines()
    number_list = []
    for line in lines:
        number_str = line.split('_')[-1]
        number = float(number_str)
        number_list.append(number)

bins = np.linspace(0, 1, 11)  # 11 edges for 10 bins

# Group the numbers and count the frequencies using numpy.histogram
counts, bin_edges = np.histogram(number_list, bins=bins)

# Create bin labels for the x-axis
bin_labels = [f"{bin_edges[i]:.1f} - {bin_edges[i+1]:.1f}" for i in range(len(bin_edges)-1)]

# Plotting the bar chart
plt.figure(figsize=(10, 6))
plt.bar(range(1, 11), counts, tick_label=bin_labels, color='skyblue', edgecolor='black')
plt.xlabel("Groups")
plt.ylabel("Frequency")
plt.title("Frequency Distribution of Numbers in Bins (0-1)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()