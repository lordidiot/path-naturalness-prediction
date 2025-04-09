import pickle

with open("features/v_deg.pkl", "rb") as f:
    data = pickle.load(f)
    
for i,j in data.items():
    print(i, j)
