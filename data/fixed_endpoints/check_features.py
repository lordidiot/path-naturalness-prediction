import pickle

with open("fixed_endpoints/science_features/v_deg.pkl", "rb") as f:
    data = pickle.load(f)

# print(len(data))
for i,j in data.items():
    print(i,j)
