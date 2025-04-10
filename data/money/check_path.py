import pickle

with open("paths.pkl", "rb") as f:
    data = pickle.load(f)

for i,j in data.items():

    print(i)
