import pickle

def save_model(model: object, filename: str) -> None:
    # save model pakai pickle
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename: str) -> object:
    # load model pakai pickle
    with open(filename, 'rb') as f:
        return pickle.load(f)