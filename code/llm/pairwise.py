from fixed_endpoints.utils import load_pickle

def main(filename: str):
    data = load_pickle(filename)
    print(len(data))


if __name__ == "__main__":
    main("../data/fixed_endpoints/science_paths_fixed_endpoints.pkl")
