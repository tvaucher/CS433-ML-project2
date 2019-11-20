import _pickle as c_pickle


def save_object(obj, filepath):
    with open(filepath, 'wb') as file:
        c_pickle.dump(obj, file)
        file.close()


def load_object(filepath):
    with open(filepath, 'rb') as file:
        obj = c_pickle.load(file)
        file.close()
        return obj
