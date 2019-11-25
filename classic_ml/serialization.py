import _pickle as c_pickle
import compress_pickle


def save_object(obj, filepath, compress=True):
    with open(filepath, 'wb') as file:
        if compress:
            compress_pickle.dump(obj, file, compression="gzip", set_default_extension=True)
        else:
            c_pickle.dump(obj, file)
        file.close()


def load_object(filepath, compressed=True):
    with open(filepath, 'rb') as file:
        if compressed:
            obj = compress_pickle.load(filepath)
        else:
            obj = c_pickle.load(file)
        file.close()
        return obj
