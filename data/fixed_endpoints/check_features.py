import pickle

with open("fixed_endpoints/money_features/v_deg.pkl", "rb") as f:
    data = pickle.load(f)
    
for i,j in data.items():
        for ls in j:
            if ls[0] < 0:
                print(i, ls)

